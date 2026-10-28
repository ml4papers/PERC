# ABPEM.py — Entropy-Calibrated Regularization (ECR) for Single-Step TTA
import torch
import torch.nn as nn
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import math


# =========================
# 1. Basic Cache and Mathematical Tools
# =========================
class Cache:
    def __init__(self, size) -> None:
        self.features, self.sample_attns, self.labels = [], [], []
        self.size = size

    def update(self): pass

    def sample_feature(self): return None


@torch.no_grad()
def js_div(p, q, eps: float = 1e-6):
    m = 0.5 * (p + q)
    return 0.5 * torch.sum(p * (p.log() - m.log()), dim=-1) + \
        0.5 * torch.sum(q * (q.log() - m.log()), dim=-1)


# =========================
# 2. Parameter Unfreezing and Management
# =========================
requires_grad_list = [
    'module.blocks_u.0.attn.qa', 'module.blocks_u.0.attn.ka', 'module.blocks_u.0.attn.va',
    'module.blocks_u.0.attn.qv', 'module.blocks_u.0.attn.kv', 'module.blocks_u.0.attn.vv'
]


def unfreeze_modality_params(model):
    model.train()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            m.requires_grad_(True)
    return model


def collect_params(model):
    params, names = [], []
    for nm, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            names.append(nm)
    return params, names


def configure_model(model):
    model.train();
    model.requires_grad_(False)
    return model


# =========================
# 3. READ TTA Wrapper (Single-Step EMA Version)
# =========================
class READ(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model, self.optimizer, self.args, self.device = model, optimizer, args, device
        self.steps = 1  # Forcefully set to 1-step Online TTA
        self.episodic = episodic
        self.scaler = GradScaler()

        # EMA Global Statistics Initialization
        self.args._ema_d_a = 0.0
        self.args._ema_d_v = 0.0
        self.args._ema_m = getattr(args, "ema_m", 0.999)  # Teacher update momentum

        # 👑 Initialize Mean Teacher Model (Freeze parameters, acts as a stable anchor)
        self.args._ema_model = deepcopy(self.model).to(device).eval()
        for p in self.args._ema_model.parameters():
            p.requires_grad_(False)

    def forward(self, x, adapt_flag, labels=None):
        if adapt_flag:
            return forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler)
        else:
            with torch.no_grad():
                self.model.eval()
                out = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode, args=self.args)
                outputs = out[0] if isinstance(out, (list, tuple)) else out
            return outputs, (torch.tensor(0.0),) * 5


# ===================================================
# 4. Core Algorithm: Single-Step ECR 👑
# ===================================================
@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler):
    core = model.module if hasattr(model, "module") else model
    ema_core = args._ema_model.module if hasattr(args._ema_model, "module") else args._ema_model
    eps = 1e-6
    a_raw, v_raw = x[0], x[1]

    # --- Step 1: Extract Stable EMA Anchor (No Gradients) ---
    with torch.no_grad():
        args._ema_model.eval()
        out_a = ema_core.forward_eval(a=a_raw, v=None, mode='audio', args=args, stage='test')
        out_v = ema_core.forward_eval(a=None, v=v_raw, mode='visual', args=args, stage='test')
        out_f = ema_core.forward_eval(a=a_raw, v=v_raw, mode=args.testmode, args=args, stage='test')

        logits_a = out_a[0] if isinstance(out_a, (list, tuple)) else out_a
        logits_v = out_v[0] if isinstance(out_v, (list, tuple)) else out_v
        logits_f = out_f[0] if isinstance(out_f, (list, tuple)) else out_f

        p_a, p_v, p_f = logits_a.softmax(-1), logits_v.softmax(-1), logits_f.softmax(-1)

        # Smoothed Batch-level Divergence Statistics
        d_a_batch = js_div(p_f, p_a).mean().item()
        d_v_batch = js_div(p_f, p_v).mean().item()

        args._ema_d_a = 0.9 * args._ema_d_a + 0.1 * d_a_batch
        args._ema_d_v = 0.9 * args._ema_d_v + 0.1 * d_v_batch

        # Calculate Continuous Divergence Weights (Temperature Softmax)
        tau = 0.1
        w_a = math.exp(-args._ema_d_a / tau)
        w_v = math.exp(-args._ema_d_v / tau)
        sum_w = w_a + w_v + eps
        w_a, w_v = w_a / sum_w, w_v / sum_w

        # Precompute Geometric Consistency Target (Teacher Target)
        log_t = torch.log_softmax(logits_a, dim=-1) * w_a + torch.log_softmax(logits_v, dim=-1) * w_v
        t_probs = torch.softmax(log_t, dim=-1).clamp_min(eps)

    # --- Step 2: Parameter Unfreezing ---
    unfreeze_modality_params(model)
    params, _ = collect_params(model)
    if len(params) > 0: optimizer.param_groups[0]['params'] = params

    # --- Step 3: Forward Pass and Multi-Constraint Regularization ---
    model.train()
    with autocast():
        out_adapt = core.forward_eval(a=a_raw, v=v_raw, mode=args.testmode, args=args, stage='adapt')
        outputs = out_adapt[0] if isinstance(out_adapt, (list, tuple)) else out_adapt
        pred_prob = outputs.softmax(dim=-1)

        # 1. Confidence-Masked Pseudo Labels (Confidence-Weighted CE)
        conf, pseudo = pred_prob.max(dim=-1)
        ce_raw = F.cross_entropy(outputs, pseudo, reduction='none')
        ce_loss = (ce_raw * conf.detach()).mean()

        # 2. Dynamic Geometric Consistency (Smoothed Geometric Consistency)
        loss_cons = torch.sum(pred_prob * (pred_prob.add(eps).log() - t_probs.log()), dim=-1).mean()

        # 3. Ranked Entropy and Class Balance (Ranked Entropy & Balance)
        p_sorted, _ = torch.sort(pred_prob, dim=-1, descending=True)
        rank = torch.arange(p_sorted.size(1), device=pred_prob.device, dtype=pred_prob.dtype)
        lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.5))
        w_rank = torch.exp(-max(1e-6, float(lam0)) * rank)
        w_rank = w_rank / (w_rank.sum() + eps)
        loss_ra = ((-p_sorted * torch.log(p_sorted + eps)) * w_rank.unsqueeze(0)).sum(dim=-1).mean()

        p_sum = pred_prob.mean(dim=0)
        loss_bal = -(p_sum * (p_sum + eps).log()).sum()

        # Combined Loss
        loss = (ce_loss * getattr(args, "ce_weight", 1.0)
                + loss_cons * getattr(args, "cons_weight", 0.4)
                + loss_ra * getattr(args, "ent_weight", 0.2)
                - loss_bal * getattr(args, "bal_weight", 0.6))

    # --- Step 4: Gradient Update ---
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    scaler.step(optimizer)
    scaler.update()

    # --- Step 5: Mean Teacher EMA Momentum Update 👑 ---
    with torch.no_grad():
        m = args._ema_m
        for param_q, param_k in zip(model.parameters(), args._ema_model.parameters()):
            if param_q.requires_grad:
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    # --- Step 6: Final Test Forward Pass ---
    model.eval()
    with torch.no_grad():
        out_final = core.forward_eval(a=a_raw, v=v_raw, mode=args.testmode, args=args, stage='test')
        outputs_final = out_final[0] if isinstance(out_final, (list, tuple)) else out_final

    return (outputs, outputs_final), (ce_loss.item(), loss_ra.item(), loss_bal.item(), loss_cons.item(), loss.item())


@torch.no_grad()
def evaluate_modality_detection(model, dataloader, device, args): return 0.0, 0.0, 0.0, 0.0

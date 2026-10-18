import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np

def compute_modality_weights_mla(logits_a, logits_v, tau: float = 1.0, eps: float = 1e-6):
    p_a = torch.softmax(logits_a, dim=1)
    p_v = torch.softmax(logits_v, dim=1)
    H_a = -(p_a * (torch.log(p_a + eps))).sum(dim=1)   # [B]
    H_v = -(p_v * (torch.log(p_v + eps))).sum(dim=1)   # [B]
    H = torch.stack([H_a, H_v], dim=1)                 # [B, 2]
    H_shift = H - H.max(dim=1, keepdim=True).values
    w = torch.softmax(-H_shift / max(tau, 1e-6), dim=1)  # [B, 2]
    q_a = w[:, 0:1]
    q_v = w[:, 1:2]
    return q_a, q_v

class Cache:
    def __init__(self, size) -> None:
        self.features = []
        self.sample_attns = []
        self.labels = []
        self.size = size
    def update(self):
        if len(self.features) > self.size:
            self.features = self.features[1:]
        if len(self.sample_attns) > self.size:
            self.sample_attns = self.sample_attns[1:]
        if len(self.labels) > self.size:
            self.labels = self.labels[1:]
    def sample_feature(self):
        idx = random.randint(0, len(self.features) - 1)
        return self.features[idx]

def kl_div(p_mean, p_std, q_mean, q_std):
    return torch.log(q_std / p_std) + (p_std ** 2 + (p_mean - q_mean) ** 2) / (2 * q_std ** 2) - 0.5

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# --------- Entropy-guided sample weighting (fusion-only) ---------
def compute_entropy_guided_weights(p_f: torch.Tensor, args):
    """
    p_f: [B,C] softmax 概率
    返回:
      w: [B] ∈ [0,1]
      info: 统计信息（可打印）
    形式: w = ((1 - H/Hmax).clamp(0,1)) ** gamma
    可选硬阈: 若 args.entw_cut in (0,1)，当 H/Hmax > entw_cut 时 w=0
    """
    eps = 1e-6
    B, C = p_f.size()
    # 熵（逐样本）& 归一化
    H = -(p_f * (p_f.clamp_min(eps)).log()).sum(dim=1)   # [B]
    Hmax = math.log(C + 1e-12)
    Hn = (H / max(Hmax, eps)).clamp(0.0, 1.0)            # [0,1]

    gamma = float(getattr(args, "entw_gamma", 1.0))
    w = (1.0 - Hn).pow(gamma)                            # 低熵→高权重
    cut = getattr(args, "entw_cut", None)
    if cut is not None:
        # 硬掩码：高于 cut 的高熵样本直接置 0
        try:
            cut = float(cut)
            if 0.0 < cut < 1.0:
                w = torch.where(Hn <= cut, w, torch.zeros_like(w))
        except Exception:
            pass
    floor = float(getattr(args, "entw_floor", 0.0))
    if floor > 0:
        w = torch.maximum(w, torch.full_like(w, floor))
    # 选项：保持 batch 期望 ~1，避免整体缩放改变优化步幅
    if getattr(args, "entw_norm_to_mean1", True):
        m = w.mean().detach()
        w = w * (1.0 / (m + eps))

    info = {
        "H_mean": float(H.mean().item()),
        "Hn_mean": float(Hn.mean().item()),
        "w_mean": float(w.mean().item()),
        "keep_ratio": float((w > 0).float().mean().item()),
        "gamma": gamma,
        "cut": (None if cut is None else float(cut)),
        "floor": floor
    }
    return w.detach(), info

# ===================== Core READ =====================

class READ(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

        self.args._ema_m = getattr(args, "ema_m", 0.999)
        self.args._ema_model = deepcopy(self.model).to(device).eval()
        for p in self.args._ema_model.parameters():
            p.requires_grad_(False)

    def forward(self, x, adapt_flag, labels=None):
        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss_tuple = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler, labels=labels)
            else:
                args = self.args
                with torch.no_grad():
                    outputs, attn_raw = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode, args=args)
                loss_tuple = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
            return outputs, loss_tuple

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    """
    ABPEM (with REM-style RE) + MLA-weighted consistency + Entropy-guided reweighting.
    Returns: (outputs_adapt, outputs_test), (ce, ra, bal, cons, total)
    """
    eps = 1e-6

    # ---------- (1) Forward (adapt; keep graph) ----------
    with autocast():
        outputs, attn_raw, cached_av_feat = model.module.forward_eval(
            a=x[0], v=x[1],
            mode=args.testmode,
            args=args,
            stage='adapt',
            labels=labels,
            ret_av_feat=True,
            output_embedding=True
        )

    # ---------- (2) MLA anchors ----------
    with torch.no_grad():
        logits_a_anchor, _ = model.module.forward_eval(a=x[0], v=None,  mode='audio',  args=args, stage='test')
        logits_v_anchor, _ = model.module.forward_eval(a=None,  v=x[1], mode='visual', args=args, stage='test')

    q_a, q_v = compute_modality_weights_mla(logits_a_anchor, logits_v_anchor, tau=getattr(args, 'mla_tau', 1.0))  # [B,1]
    gap_val = float((q_a - q_v).abs().mean().item())

    # ---------- (3) Probabilities ----------
    p_f = torch.softmax(outputs, dim=-1)                  # [B,C]
    p_a = torch.softmax(logits_a_anchor, dim=-1)
    p_v = torch.softmax(logits_v_anchor, dim=-1)

    # ---------- (4) Entropy-guided weights (fusion only) ----------
    w_ent, info_ent = compute_entropy_guided_weights(p_f.detach(), args)  # [B]

    # ---------- (5) Losses (per-sample then weighted) ----------
    # 5.1 REM: per-sample ranked entropy
    p_sorted, _ = torch.sort(p_f, dim=-1, descending=True)         # [B, C]
    C = p_sorted.size(1)
    rank = torch.arange(C, device=p_sorted.device, dtype=p_sorted.dtype)
    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam1 = getattr(args, "re_lambda1", 0.0)
    lam_eff = max(1e-6, float(lam0 + lam1 * gap_val))
    w_rank = torch.exp(-lam_eff * rank); w_rank = w_rank / (w_rank.sum() + 1e-12)
    ent_sorted = -p_sorted * torch.log(p_sorted + eps)             # [B,C]
    ra_vec = (ent_sorted * w_rank.unsqueeze(0)).sum(dim=-1)        # [B]
    loss_ra = (w_ent * ra_vec).sum() / (w_ent.sum() + eps)

    # 5.2 Balancing（batch 级，不加权）
    if getattr(args, "fix_loss_bal", False):
        p_mean = p_f.mean(dim=-2)  # [C]
        loss_bal = - (p_mean * torch.log(p_mean + eps)).sum()
    else:
        p_sum = p_f.sum(dim=-2)    # [C]
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()

    # 5.3 Pseudo CE: per-sample → weighted
    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_vec = torch.nn.functional.cross_entropy(outputs, pseudo, reduction='none')  # [B]
    ce_loss = (w_ent * ce_vec).sum() / (w_ent.sum() + eps)

    # 5.4 Consistency: per-sample → weighted
    log_pf = torch.log(p_f + eps)
    kl_f_a = (p_f * (log_pf - torch.log(p_a + eps))).sum(dim=1)  # [B]
    kl_f_v = (p_f * (log_pf - torch.log(p_v + eps))).sum(dim=1)  # [B]
    cons_vec = (q_a.squeeze(1) * kl_f_a + q_v.squeeze(1) * kl_f_v)  # [B]
    loss_cons = (w_ent * cons_vec).sum() / (w_ent.sum() + eps)

    # ---------- (6) Total ----------
    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
            + loss_ra * getattr(args, "ent_weight", 0.0)
            - loss_bal * getattr(args, "bal_weight", 0.0)
            + loss_cons * getattr(args, "cons_weight", 1.0))

    # ---------- (7) Backprop / step ----------
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0)
    scaler.step(optimizer)
    scaler.update()

    # ---------- (8) Report ----------
    with torch.no_grad():
        with autocast():
            outputs2, _ = model.module.forward_eval(
                a=x[0], v=x[1], mode=args.testmode, args=args,
                stage='test', labels=labels, av_feat=cached_av_feat
            )

    if getattr(args, "debug_entw", False):
        print(f"[ENTW] Hn_mean={info_ent['Hn_mean']:.3f}, w_mean={info_ent['w_mean']:.3f}, "
              f"keep={info_ent['keep_ratio']:.2f}, gamma={info_ent['gamma']}, "
              f"cut={info_ent['cut']}, floor={info_ent['floor']}")

    return (outputs, outputs2), (
        ce_loss.item(),
        loss_ra.item(),
        loss_bal.item(),
        (loss_cons.item() if isinstance(loss_cons, torch.Tensor) else float(loss_cons)),
        loss.item()
    )

# ===================== Trainable submodules & utils =====================

requires_grad_list = [
    'module.blocks_u.0.attn.qa',
    'module.blocks_u.0.attn.ka',
    'module.blocks_u.0.attn.va',
    'module.blocks_u.0.attn.qv',
    'module.blocks_u.0.attn.kv',
    'module.blocks_u.0.attn.vv'
]

def collect_params(model):
    params_fusion_qkv = []
    names_fusion_qkv = []
    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")
    return params_fusion_qkv, names_fusion_qkv

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            print(f"[INFO] Unfreezing module: {nm}")
            for param in m.parameters():
                param.requires_grad = True
    return model



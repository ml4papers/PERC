
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np


def compute_modality_weights_qmf(logits_a, logits_v, temperature=0.1):
    """
    QMF: 基于模态质量（置信度）的权重计算
    (Kept for backward-compatibility; recommend using compute_modality_weights_mla)
    """
    conf_a = torch.softmax(logits_a, dim=1).max(dim=1)[0]  # [B]
    conf_v = torch.softmax(logits_v, dim=1).max(dim=1)[0]  # [B]
    quality_stack = torch.stack([conf_a, conf_v], dim=1)   # [B, 2]
    quality_stack = quality_stack / max(temperature, 1e-6)
    weights = torch.softmax(quality_stack, dim=1)          # [B, 2]
    q_a = weights[:, 0].unsqueeze(1)                       # [B, 1]
    q_v = weights[:, 1].unsqueeze(1)                       # [B, 1]
    return q_a, q_v


def compute_modality_weights_mla(logits_a, logits_v, tau: float = 1.0, eps: float = 1e-6):
    """
    MLA: 基于每模态预测熵的动态权重（softmax over -entropy）。
    返回逐样本的模态权重 q_a,q_v，且 q_a + q_v = 1。
    """
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


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    """
    ABPEM (with REM-style RE) + MLA-weighted consistency + Adaptive Rank Weighting.
    - No OGM/GE.
    - Returns: (outputs_adapt, outputs_test), (re, bal, attn, cons, total)
    """
    # ---------- (1) Forward (adapt; keep graph) ----------
    with autocast():
        outputs, attn_raw, cached_av_feat = model.module.forward_eval(
            a=x[0], v=x[1],
            mode=args.testmode,      # 'multimodal'
            args=args,
            stage='adapt',
            labels=labels,
            ret_av_feat=True,
            output_embedding=True
        )

    # ---------- (2) MLA weights first (for adaptive rank + consistency) ----------
    with torch.no_grad():
        logits_a_anchor, _ = model.module.forward_eval(
            a=x[0], v=None, mode='audio',  args=args, stage='test'
        )
        logits_v_anchor, _ = model.module.forward_eval(
            a=None,  v=x[1], mode='visual', args=args, stage='test'
        )

    q_a, q_v = compute_modality_weights_mla(
        logits_a_anchor, logits_v_anchor, tau=getattr(args, 'mla_tau', 1.0)
    )  # [B,1] each
    # batch-level modality dominance gap
    gap_val = float((q_a - q_v).abs().mean().item())

    # ---------- (3) Core losses ----------
    # REM: Ranked Entropy with ADAPTIVE rank decay
    pred = outputs.softmax(dim=-1)                                  # [B, C]
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)         # [B, C]
    C = p_sorted.size(1)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)

    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam1 = getattr(args, "re_lambda1", 0.)   # default 0 keeps backward-compatibility
    lam_eff = max(1e-6, float(lam0 + lam1 * gap_val))               # scalar
    w = torch.exp(-lam_eff * rank); w = w / (w.sum() + 1e-12)       # [C]

    ent_sorted = -p_sorted * torch.log(p_sorted + 1e-6)
    loss_ra = (ent_sorted * w.unsqueeze(0)).sum(dim=-1).mean()

    # Balancing
    if getattr(args, "fix_loss_bal", False):
        p_sum = outputs.softmax(dim=-1).mean(dim=-2)
        loss_bal = - (p_sum * p_sum.log()).sum()
    else:
        p_sum = outputs.softmax(dim=-1).sum(dim=-2)
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()

    # Attention bootstrapping (KLD of attention stats)
    attn_raw_a2a = attn_raw[:, :512, :512]
    attn_raw_v2v = attn_raw[:, 512:, 512:]
    attn_raw_a2v = attn_raw[:, :512, 512:]
    attn_raw_v2a = attn_raw[:, 512:, :512]

    def _kld_norm(p_mean, p_std, q_mean, q_std):
        return torch.log(q_std / p_std) + (p_std ** 2 + (p_mean - q_mean) ** 2) / (2 * q_std ** 2) - 0.5

    a2a_mean, a2a_std = attn_raw_a2a.mean((1,2)).detach(), attn_raw_a2a.std((1,2)).detach()
    v2v_mean, v2v_std = attn_raw_v2v.mean((1,2)).detach(), attn_raw_v2v.std((1,2)).detach()
    a2v_mean, a2v_std = attn_raw_a2v.mean((1,2)),            attn_raw_a2v.std((1,2))
    v2a_mean, v2a_std = attn_raw_v2a.mean((1,2)),            attn_raw_v2a.std((1,2))
    loss_attn = _kld_norm(v2a_mean, v2a_std, a2a_mean, a2a_std).mean() \
              + _kld_norm(a2v_mean, a2v_std, v2v_mean, v2v_std).mean()

    # Small pseudo-CE
    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = torch.nn.functional.cross_entropy(outputs, pseudo)

    # ---------- (4) MLA-weighted consistency ----------
    eps = 1e-6
    p_f = pred
    p_a = torch.softmax(logits_a_anchor, dim=1)
    p_v = torch.softmax(logits_v_anchor, dim=1)

    conf_th = getattr(args, 'cons_conf_th', 0.0)
    if conf_th > 0.0:
        conf = p_f.max(dim=1).values
        mask = (conf >= conf_th).float().unsqueeze(1)  # [B,1]
    else:
        mask = 1.0

    log_pf = torch.log(p_f + eps)
    kl_f_a = (p_f * (log_pf - torch.log(p_a + eps))).sum(dim=1, keepdim=True)  # [B,1]
    kl_f_v = (p_f * (log_pf - torch.log(p_v + eps))).sum(dim=1, keepdim=True)  # [B,1]

    cons_weight = getattr(args, 'cons_weight', 0.5)
    loss_cons = cons_weight * ((q_a * mask * kl_f_a).mean() + (q_v * mask * kl_f_v).mean())

    # ---------- (5) Total ----------
    loss = (ce_loss
            + loss_ra * args.ent_weight
            - loss_bal * args.bal_weight
            + loss_attn * args.attn_weight
            + loss_cons)

    # ---------- (6) Backprop / step ----------
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0
    )
    scaler.step(optimizer)
    scaler.update()

    # ---------- (7) Second forward (report) ----------
    with torch.no_grad():
        with autocast():
            outputs2, _ = model.module.forward_eval(
                a=x[0], v=x[1], mode=args.testmode, args=args,
                stage='test', labels=labels, av_feat=cached_av_feat
            )

    return (outputs, outputs2), (
        ce_loss.item(),
        loss_ra.item(),
        loss_bal.item(),
        loss_attn.item(),
        (loss_cons.item() if isinstance(loss_cons, torch.Tensor) else float(loss_cons)),
        loss.item()
    )


requires_grad_list = [
    'module.blocks_u.0.attn.qa',
    'module.blocks_u.0.attn.ka',
    'module.blocks_u.0.attn.va',
    'module.blocks_u.0.attn.qv',
    'module.blocks_u.0.attn.kv',
    'module.blocks_u.0.attn.vv'
    'module.blocks_u.0.norm1',  # 新增：LayerNorm
    'module.blocks_u.0.norm2',  # 新增：LayerNorm
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

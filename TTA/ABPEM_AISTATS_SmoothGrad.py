
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
    # ---------- (0) Input noise (Dn2) ----------
    a_in, v_in = x[0], x[1]
    use_dn2 = bool(getattr(args, 'aug_noise', False))
    if use_dn2:
        # 为保持可复现，噪声只在该函数内采样
        if v_in is not None:
            v_noise = torch.randn_like(v_in) * float(getattr(args, 'noise_v', 0.03))
            v_noisy = v_in + v_noise
        else:
            v_noisy = None
        if a_in is not None:
            a_noise = torch.randn_like(a_in) * float(getattr(args, 'noise_a', 0.01))
            a_noisy = a_in + a_noise
        else:
            a_noisy = None
    else:
        v_noisy, a_noisy = v_in, a_in

    # ---------- (1) Forward (adapt; keep graph) ----------
    with autocast():
        # clean view
        outputs_c, attn_raw, cached_av_feat = model.module.forward_eval(
            a=a_in, v=v_in, mode=args.testmode, args=args,
            stage='adapt', labels=labels, ret_av_feat=True, output_embedding=True
        )
        if use_dn2:
            # noisy view
            outputs_n, _attn_dummy, _ = model.module.forward_eval(
                a=a_noisy, v=v_noisy, mode=args.testmode, args=args,
                stage='adapt', labels=labels, ret_av_feat=False, output_embedding=False
            )
            # 对数空间平均更稳（避免过度拉平）
            outputs = torch.logsumexp(torch.stack([outputs_c, outputs_n], dim=0), dim=0) - math.log(2.0)
        else:
            outputs = outputs_c

    # ---------- (2) MLA weights first (for adaptive rank + consistency) ----------
    with torch.no_grad():
        logits_a_anchor, _ = model.module.forward_eval(a=a_in, v=None, mode='audio', args=args, stage='test')
        logits_v_anchor, _ = model.module.forward_eval(a=None, v=v_in, mode='visual', args=args, stage='test')
        if use_dn2:
            logits_a_anchor_n, _ = model.module.forward_eval(a=a_noisy, v=None, mode='audio', args=args, stage='test')
            logits_v_anchor_n, _ = model.module.forward_eval(a=None, v=v_noisy, mode='visual', args=args, stage='test')
            logits_a_anchor = torch.logsumexp(torch.stack([logits_a_anchor, logits_a_anchor_n], dim=0),
                                              dim=0) - math.log(2.0)
            logits_v_anchor = torch.logsumexp(torch.stack([logits_v_anchor, logits_v_anchor_n], dim=0),
                                              dim=0) - math.log(2.0)

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

    # ------------------Small pseudo-CE--------------------------
    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = torch.nn.functional.cross_entropy(outputs, pseudo)

    # --- EMA update ---
    # ema_model = args._ema_model
    # m = args._ema_m
    # with torch.no_grad():
    #     for p_s, p_t in zip(model.parameters(), ema_model.parameters()):
    #         p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
    #
    # # --- Teacher forward (no grad) ---
    # teacher = ema_model.module if hasattr(ema_model, "module") else ema_model
    # with torch.no_grad(), autocast():
    #     logits_t, _ = teacher.forward_eval(
    #         a=x[0], v=x[1], mode=args.testmode, args=args,
    #         stage='test', labels=None, av_feat=cached_av_feat
    #     )
    #
    # # --- Soft CE (fp32, no mask) ---
    # p_t = torch.softmax((logits_t).float(), dim=1)  # [B,C], fp32
    # log_p_s = torch.log_softmax((outputs).float(), dim=1)  # [B,C], fp32
    # ce_loss = -(p_t * log_p_s).sum(dim=1).mean()  # Hinton scaling

    # ---------- (4) MLA-weighted consistency ----------
    # eps = 1e-6
    # p_f = pred
    # p_a = torch.softmax(logits_a_anchor, dim=1)
    # p_v = torch.softmax(logits_v_anchor, dim=1)
    #
    # if args.cons_conf_th > 0.0:
    #     conf = p_f.max(dim=1).values
    #     mask = (conf >= args.cons_conf_th).float().unsqueeze(1)  # [B,1]
    # else:
    #     mask = 1.0
    #
    # log_pf = torch.log(p_f + eps)
    # kl_f_a = (p_f * (log_pf - torch.log(p_a + eps))).sum(dim=1, keepdim=True)  # [B,1]
    # kl_f_v = (p_f * (log_pf - torch.log(p_v + eps))).sum(dim=1, keepdim=True)  # [B,1]
    #
    # loss_cons = ((q_a * mask * kl_f_a).mean() + (q_v * mask * kl_f_v).mean())

    # ----------OR (4) PoE Consistency (替换原来的 MLA consistency) ----------
    eps = 1e-6
    p_f = pred  # fused probs, [B,C]

    # unimodal probs
    p_a = torch.softmax(logits_a_anchor, dim=1).clamp_min(eps)  # [B,C]
    p_v = torch.softmax(logits_v_anchor, dim=1).clamp_min(eps)  # [B,C]

    # 置信度自适应：用“归一化熵”的反比来给 gamma（越确定 -> 权重大）
    # H_norm in [0,1]; r = 1/(H_norm+eps); gamma = r_a/(r_a+r_v)
    with torch.no_grad():
        C = p_f.size(1)
        logC = math.log(C + 1e-12)

        Ha = -(p_a * p_a.log()).sum(dim=1) / logC          # [B]
        Hv = -(p_v * p_v.log()).sum(dim=1) / logC          # [B]
        ra = 1.0 / (Ha + 1e-6)
        rv = 1.0 / (Hv + 1e-6)
        gamma = (ra / (ra + rv)).unsqueeze(1)              # [B,1]

    # Product-of-Experts 目标（几何平均：log 空间线性组合）
    log_t = gamma * p_a.log() + (1.0 - gamma) * p_v.log()  # [B,C]
    t = torch.softmax(log_t, dim=1).clamp_min(eps)         # [B,C]

    # 只需一个目标，避免与原 MLA 的“双 KL”重叠
    # 你可以用 CE 或 KL，这里用 KL(p_f || t)
    loss_cons = torch.sum(p_f * (p_f.add(eps).log() - t.log()), dim=1).mean()


    # ---------- (5) Total ----------
    loss = (ce_loss * args.ce_weight
            + loss_ra * args.ent_weight
            - loss_bal * args.bal_weight
            + loss_cons * args.cons_weight)

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


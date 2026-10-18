# -*- coding: utf-8 -*-
# Integrated ABPEM (Selective QKV FT + One-shot corruption decision + Diffusion denoise stubs)
import math
import random
import torch
import torch.nn as nn
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler

# =========================
# 0) Utilities & constants
# =========================

A_QKV = ["blocks_u.0.attn.qa", "blocks_u.0.attn.ka", "blocks_u.0.attn.va"]
V_QKV = ["blocks_u.0.attn.qv", "blocks_u.0.attn.kv", "blocks_u.0.attn.vv"]

def _strip_module_prefix(nm: str) -> str:
    return nm[7:] if nm.startswith("module.") else nm

def _name_hit(nm: str, keys) -> bool:
    nm2 = _strip_module_prefix(nm)
    return any(nm2.endswith(k) for k in keys)

def _softmax_stable(logits, eps=1e-6):
    return torch.softmax(logits.float(), dim=1).clamp_min(eps)

def _kl_pq(p, q, eps=1e-6):
    # KL(p||q) = sum p (log p - log q)
    return torch.sum(p * (p.add(eps).log() - q.add(eps).log()), dim=1)

def compute_modality_weights_mla(logits_a, logits_v, tau: float = 1.0, eps: float = 1e-6):
    """基于熵的 MLA 权重（softmax over -entropy），返回 [B,1] 的 q_a, q_v."""
    p_a = torch.softmax(logits_a, dim=1)
    p_v = torch.softmax(logits_v, dim=1)
    H_a = -(p_a * (torch.log(p_a + eps))).sum(dim=1)
    H_v = -(p_v * (torch.log(p_v + eps))).sum(dim=1)
    H = torch.stack([H_a, H_v], dim=1)
    H_shift = H - H.max(dim=1, keepdim=True).values
    w = torch.softmax(-H_shift / max(tau, 1e-6), dim=1)
    return w[:, 0:1], w[:, 1:2]

class Cache:
    def __init__(self, size) -> None:
        self.features, self.sample_attns, self.labels = [], [], []
        self.size = size
    def update(self):
        while len(self.features) > self.size:      self.features.pop(0)
        while len(self.sample_attns) > self.size:  self.sample_attns.pop(0)
        while len(self.labels) > self.size:        self.labels.pop(0)
    def sample_feature(self):
        idx = random.randint(0, len(self.features) - 1)
        return self.features[idx]

def kl_div(p_mean, p_std, q_mean, q_std):
    return torch.log(q_std / p_std) + (p_std ** 2 + (p_mean - q_mean) ** 2) / (2 * q_std ** 2) - 0.5


# ===================================
# 1) Diffusion denoise (plug-in API)
# ===================================

class AudioDiffusionDenoiser:
    """把你的音频扩散模型接到 model 上；没提供时自动直通。"""
    def __init__(self, model=None):
        self.model = model
    @torch.no_grad()
    def __call__(self, a: torch.Tensor, args) -> torch.Tensor:
        if self.model is None:
            return a
        # 你的推理逻辑（确保输出与 a 同形状/设备/类型）
        return self.model.denoise(a)

class VisualDiffusionDenoiser:
    """把你的视觉扩散模型接到 model 上；没提供时自动直通。"""
    def __init__(self, model=None):
        self.model = model
    @torch.no_grad()
    def __call__(self, v: torch.Tensor, args) -> torch.Tensor:
        if self.model is None:
            return v
        return self.model.denoise(v)


# =========================================
# 2) Corruption decision (one-shot, EMA-K)
# =========================================

def update_corruption_decision_once(core, x, args):
    """
    用前 K 个样本的 KL 差 δ=KL(p_f||p_v)-KL(p_f||p_a) 的 EMA 来一次性判定:
      ema >  gap_thr → 'video'
      ema < -gap_thr → 'audio'
      否则 → 'none'
    固定后写入 args._corr_fixed ∈ {audio, video, none}。
    """
    if getattr(args, "_corr_fixed", None) is not None:
        return

    ema_m   = float(getattr(args, "corr_ema_m", 0.9))
    warm_k  = int(getattr(args, "corr_warmup", 100))
    gap_thr = float(getattr(args, "corr_gap_thr", 0.02))

    with torch.no_grad():
        logits_f, _ = core.forward_eval(a=x[0], v=x[1], mode=args.testmode, args=args, stage='test')
        logits_a, _ = core.forward_eval(a=x[0], v=None,  mode='audio',  args=args, stage='test')
        logits_v, _ = core.forward_eval(a=None,  v=x[1], mode='visual', args=args, stage='test')

    p_f = _softmax_stable(logits_f)
    p_a = _softmax_stable(logits_a)
    p_v = _softmax_stable(logits_v)

    kl_f_a = _kl_pq(p_f, p_a).mean().item()
    kl_f_v = _kl_pq(p_f, p_v).mean().item()
    delta  = kl_f_v - kl_f_a  # >0 → visual 更偏离融合

    cnt = int(getattr(args, "_corr_cnt", 0))
    ema = float(getattr(args, "_corr_ema", 0.0))
    ema = ema_m * ema + (1.0 - ema_m) * delta
    cnt += p_f.size(0)

    args._corr_ema = ema
    args._corr_cnt = cnt

    if cnt >= warm_k:
        if   ema >  gap_thr: args._corr_fixed = "video"
        elif ema < -gap_thr: args._corr_fixed = "audio"
        else:                args._corr_fixed = "none"
        if getattr(args, "debug_decide", False):
            print(f"[DECIDE/FIXED] emaΔ={ema:.4f} after {cnt} → {args._corr_fixed}")


# ===========================================
# 3) Freeze/Unfreeze: only fusion-layer QKV
# ===========================================

def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_qkv_by_corruption(model: nn.Module, corr: str):
    """ 只解冻被判定腐败模态的融合层 Q/K/V。 """
    targets = A_QKV if corr == "audio" else V_QKV
    n = 0
    for nm, m in model.named_modules():
        if _name_hit(nm, targets):
            for p in m.parameters():
                p.requires_grad = True
            n += sum(p.numel() for p in m.parameters())
            print(f"[FT] unfreeze {nm}")
    print(f"[FT] total trainable params ≈ {n/1e6:.3f} M")

def collect_qkv_params(model: nn.Module):
    params, names = [], []
    keys = A_QKV + V_QKV
    for nm, m in model.named_modules():
        if _name_hit(nm, keys):
            for pn, p in m.named_parameters():
                if pn in ("weight", "bias") and p.requires_grad:
                    params.append(p); names.append(f"{nm}.{pn}")
    return params, names


# =========================
# 4) READ wrapper (unchanged)
# =========================

class READ(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

        # EMA teacher（当前未使用；保留接口）
        self.args._ema_m = getattr(args, "ema_m", 0.999)
        self.args._ema_model = deepcopy(self.model).to(device).eval()
        for p in self.args._ema_model.parameters():
            p.requires_grad_(False)

    def forward(self, x, adapt_flag, labels=None):
        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss_tuple = forward_and_adapt(
                    x, self.model, self.optimizer, self.args, self.scaler, labels=labels
                )
            else:
                args = self.args
                with torch.no_grad():
                    outputs, attn_raw = self.model.module.forward_eval(
                        a=x[0], v=x[1], mode=self.args.testmode, args=args
                    )
                loss_tuple = (torch.tensor(0.0),)*5
            return outputs, loss_tuple


# ==================================
# 5) The main forward_and_adapt core
# ==================================

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    """
    流程：
      1) 用前 K 个样本的 KL-EMA 一次性固定腐败模态（audio/video/none）；
      2) 固定后：只解冻对应侧融合层 QKV；（none 则全冻）
      3) 对判定腐败的单模态做扩散去噪（若提供 denoiser），得到 a~ / v~；
      4) 以去噪单模态 anchor 构造 MLA-consistency 目标 t，和 REM 一起更新；
      5) warmup 未完成时仅推理、零损失，不做 scaler/step。
    """
    core = model.module if hasattr(model, "module") else model
    device = next(core.parameters()).device
    eps = 1e-6

    # ---- (0) 更新一次腐败判定（只在前 warmup 内生效） ----
    update_corruption_decision_once(core, x, args)
    corr = getattr(args, "_corr_fixed", None)

    # ---- (1) 未固定 → 只推理 + 零损失（不做任何优化步骤）----
    if corr is None:
        with torch.no_grad(), autocast():
            out = core.forward_eval(
                a=x[0], v=x[1], mode=args.testmode, args=args,
                stage='test', labels=labels, ret_av_feat=False, output_embedding=False
            )
        logits = out[0] if isinstance(out, (list, tuple)) else out
        z = torch.tensor(0.0, device=device)
        return (logits, logits), (z.item(),)*5

    # ---- (2) 固定后，按判定设置可训练集合（只执行一次）----
    if not getattr(args, "_ft_applied", False):
        freeze_all(model)
        if corr in ("audio", "video"):
            unfreeze_qkv_by_corruption(model, corr)
        args._ft_applied = True

        # 重建优化器参数组：只含当前 requires_grad=True 的 QKV
        params, _ = collect_qkv_params(model)
        optimizer.param_groups.clear()
        if len(params) > 0:
            optimizer.add_param_group({
                "params": params,
                "lr": getattr(args, "lr", 1e-4),
                "weight_decay": getattr(args, "wd", 0.0),
            })
        if getattr(args, "debug_decide", False):
            ntr = sum(p.numel() for p in params)
            print(f"[OPT] rebuild optimizer with {ntr/1e6:.3f} M trainables; corr={corr}")

    # ---- (3) 得到（可能去噪的）单模态 anchors ----
    a_in, v_in = x[0], x[1]
    a_tilde, v_tilde = a_in, v_in

    if corr == "audio" and getattr(args, "use_audio_denoise", True):
        denoiser = getattr(args, "audio_denoiser", AudioDiffusionDenoiser(None))
        a_tilde = denoiser(a_in, args)

    if corr == "video" and getattr(args, "use_visual_denoise", True):
        denoiser = getattr(args, "visual_denoiser", VisualDiffusionDenoiser(None))
        v_tilde = denoiser(v_in, args)

    with torch.no_grad():
        logits_a_tilde, _ = core.forward_eval(a=a_tilde, v=None,  mode='audio',  args=args, stage='test')
        logits_v_tilde, _ = core.forward_eval(a=None,   v=v_tilde, mode='visual', args=args, stage='test')

    # ---- (4) 带图前向（adapt-stage）----
    with autocast():
        out = core.forward_eval(
            a=a_in, v=v_in, mode=args.testmode, args=args,
            stage='adapt', labels=labels, ret_av_feat=True, output_embedding=True
        )
    if isinstance(out, (list, tuple)) and len(out) == 3:
        outputs, attn_raw, cached_av_feat = out
    else:
        outputs, attn_raw = out
        cached_av_feat = None

    # ---- (5) 损失：REM + MLA-consistency（用去噪 anchors）----
    pred = torch.softmax(outputs, dim=1)  # p_f [B,C]
    C = pred.size(1)
    logC = math.log(C + 1e-12)

    # MLA consistency target
    p_a = _softmax_stable(logits_a_tilde, eps)
    p_v = _softmax_stable(logits_v_tilde, eps)
    with torch.no_grad():
        Ha = -(p_a * p_a.log()).sum(dim=1) / logC
        Hv = -(p_v * p_v.log()).sum(dim=1) / logC
        ra = 1.0 / (Ha + 1e-6)
        rv = 1.0 / (Hv + 1e-6)
        gamma = (ra / (ra + rv)).unsqueeze(1)  # [B,1]
    log_t = gamma * p_a.log() + (1.0 - gamma) * p_v.log()
    t = torch.softmax(log_t, dim=1).clamp_min(eps)
    loss_cons = torch.sum(pred * (pred.add(eps).log() - t.log()), dim=1).mean()

    # REM（ranked entropy with optional adaptivity）
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)
    lam0 = float(getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.5)))
    lam1 = float(getattr(args, "re_lambda1", 0.0))
    gap_val = float((Ha - Hv).abs().mean().item())  # 可选的自适应项
    lam_eff = max(1e-6, lam0 + lam1 * gap_val)
    w = torch.exp(-lam_eff * rank); w = w / (w.sum() + 1e-12)
    ent_sorted = -p_sorted * torch.log(p_sorted + 1e-6)
    loss_ra = (ent_sorted * w.unsqueeze(0)).sum(dim=-1).mean()

    # Balancing（按需保留；设权重为 0 可关闭）
    if getattr(args, "fix_loss_bal", False):
        p_sum = pred.mean(dim=-2)
        loss_bal = - (p_sum * p_sum.log()).sum()
    else:
        p_sum = pred.sum(dim=-2)
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()

    # 伪 CE（弱约束）
    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = torch.nn.functional.cross_entropy(outputs, pseudo)

    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
            + loss_ra * getattr(args, "ent_weight", 1.0)
            - loss_bal * getattr(args, "bal_weight", 0.0)
            + loss_cons * getattr(args, "cons_weight", 0.2))

    # ---- (6) Step ----
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0
    )
    scaler.step(optimizer)
    scaler.update()

    # ---- (7) 报告前向（test-stage）----
    with torch.no_grad(), autocast():
        out2 = core.forward_eval(
            a=a_in, v=v_in, mode=args.testmode, args=args,
            stage='test', labels=labels, av_feat=cached_av_feat
        )
    outputs2 = out2[0] if isinstance(out2, (list, tuple)) else out2

    return (outputs, outputs2), (
        ce_loss.item(),
        loss_ra.item(),
        loss_bal.item(),
        (loss_cons.item() if isinstance(loss_cons, torch.Tensor) else float(loss_cons)),
        loss.item()
    )


# ===========================
# 6) Configure & collectors
# ===========================

def configure_model(model):
    """
    初始**全冻结**。不再提前解冻 QKV；
    等 corruption 判定固定后，在 forward_and_adapt 内部调用 unfreeze_qkv_by_corruption()。
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    return model

def collect_params(model):
    """
    与主脚本兼容的占位：返回当前已 requires_grad=True 的 QKV 参数。
    """
    return collect_qkv_params(model)







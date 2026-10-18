import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================== DUDA: Visual branch ===========================

def _fft_lowpass_2d(x: torch.Tensor, cutoff_frac: float = 0.22, ramp: float = 0.15):
    """
    x: [B,C,H,W], float tensor in any range. Returns low-passed x (real).
    cutoff_frac in (0,0.5): pass low freq within radius (cutoff*R), cosine ramp to 0.
    """
    if cutoff_frac <= 0:
        return x
    B, C, H, W = x.shape
    X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))

    yy = torch.linspace(-1, 1, H, device=x.device).view(H, 1).expand(H, W)
    xx = torch.linspace(-1, 1, W, device=x.device).view(1, W).expand(H, W)
    rr = torch.sqrt(xx*xx + yy*yy)  # [H,W], center=0

    r0 = cutoff_frac
    r1 = min(0.5, cutoff_frac + ramp)
    mask_core = (rr <= r0).float()
    # cosine ramp from r0->r1
    ramp_zone = (rr > r0) & (rr < r1)
    mask = torch.zeros_like(rr)
    mask = mask + mask_core
    if ramp_zone.any():
        t = (rr[ramp_zone] - r0) / (r1 - r0 + 1e-8)
        mask[ramp_zone] = 0.5*(1.0 + torch.cos(math.pi * t))  # smooth down to 0
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    X_lp = X * mask
    x_lp = torch.fft.ifft2(torch.fft.ifftshift(X_lp, dim=(-2, -1)), dim=(-2, -1)).real
    return x_lp

def _gaussian_blur_2d(x: torch.Tensor, sigma: float):
    """
    lightweight separable gaussian blur; sigma in pixels.
    """
    if sigma <= 0:
        return x
    radius = int(max(1, round(3*sigma)))
    ksize = 2*radius + 1
    # 1D kernel
    t = torch.arange(-radius, radius+1, device=x.device, dtype=x.dtype)
    g = torch.exp(-0.5*(t/sigma)**2)
    g = g / (g.sum()+1e-8)
    g = g.view(1,1,ksize)
    # depthwise conv
    B,C,H,W = x.shape
    x = F.conv2d(x, g.expand(C,1,ksize).unsqueeze(3), padding=(radius,0), groups=C)
    x = F.conv2d(x, g.expand(C,1,ksize).unsqueeze(2), padding=(0,radius), groups=C)
    return x

def _tv_smooth_step(x: torch.Tensor, lam: float = 0.0):
    """
    single ROF-like step via Laplacian (very light). lam ~ [0, 0.2]
    """
    if lam <= 0:
        return x
    # 3x3 Laplacian kernel
    lap = torch.tensor([[0, 1, 0],
                        [1,-4, 1],
                        [0, 1, 0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    B,C,H,W = x.shape
    y = x - lam*F.conv2d(x, lap.expand(C,1,3,3), padding=1, groups=C)
    return y

def duda_denoise_visual(x: torch.Tensor, args):
    """
    x: [B,C,H,W].  (called on the MODALITY INPUT TENSOR for anchor)
    Steps: FFT low-pass -> annealed Gaussian smoothing -> light TV.
    """
    # (1) low-pass
    x1 = _fft_lowpass_2d(x, cutoff_frac=float(getattr(args,'duda_v_cut',0.22)),
                            ramp=float(getattr(args,'duda_v_ramp',0.15)))
    # (2) annealed blur
    steps  = int(getattr(args,'duda_v_steps',2))
    sigma0 = float(getattr(args,'duda_v_sigma0',0.8))
    gamma  = float(getattr(args,'duda_v_gamma',0.5))
    y = x1
    for t in range(steps):
        y = _gaussian_blur_2d(y, sigma=sigma0*(gamma**t))
    # (3) light TV
    tvw = float(getattr(args,'duda_v_tv',0.0))
    y = _tv_smooth_step(y, lam=tvw)
    return y

# =========================== DUDA: Audio (spectrogram) ===========================

import torch.nn.functional as F  # 确保这里用的是 F（不要被局部变量覆盖）

# ---------- 形状规范化工具：把任意谱张量转成 [B,1,F,T]，并提供还原函数 ----------
def _to_b1ft(S: torch.Tensor):
    """
    输入 S 可能是 [B, F, T] 或 [B, T, F] 或 [B, 1, F, T]。
    返回：S4=[B,1,F,T]，back() 用于把处理后的结果还原回原始形状。
    """
    if S.ndim == 4:  # 已是 [B,1,F,T] 或 [B,C,F,T]（但我们期望 C=1）
        if S.size(1) == 1:
            return S, (lambda X: X)  # 无需还原
        else:
            # 若出现 C!=1，按第 1 维聚合到单通道（极少见，保守处理）
            S4 = S.mean(dim=1, keepdim=True)
            def back(X):  # 复制回原通道数（简单重复）
                return X.expand(-1, S.size(1), -1, -1)
            return S4, back

    if S.ndim == 3:
        B, D1, D2 = S.shape
        # 假设时间长度通常 > 频率（如 T≈1024, F≈128）
        if D2 >= D1:   # [B, F, T]
            S4 = S.unsqueeze(1)  # -> [B,1,F,T]
            def back(X): return X.squeeze(1)
        else:          # [B, T, F]
            S4 = S.transpose(1, 2).unsqueeze(1)  # -> [B,1,F,T]
            def back(X): return X.squeeze(1).transpose(1, 2)
        return S4, back

    raise ValueError(f"Unexpected spectrogram shape: {tuple(S.shape)}")


def _conv1d_along_time(S: torch.Tensor, k: int):
    # S: [B,1,F,T]; 沿 T 做均值卷积
    if k <= 1:
        return S
    k = k if (k % 2 == 1) else (k + 1)
    ker = torch.ones(1, 1, 1, k, device=S.device, dtype=S.dtype) / k  # [1,1,1,k]
    S2 = F.conv2d(S, ker, padding=(0, k // 2))
    return S2


def _conv1d_along_freq(S: torch.Tensor, k: int):
    # S: [B,1,F,T]; 沿 F 做均值卷积
    if k <= 1:
        return S
    k = k if (k % 2 == 1) else (k + 1)
    ker = torch.ones(1, 1, k, 1, device=S.device, dtype=S.dtype) / k  # [1,1,k,1]
    S2 = F.conv2d(S, ker, padding=(k // 2, 0))
    return S2


def _spectral_gate(S: torch.Tensor, ratio: float = 0.25):
    """
    简易谱门控：与时间中位数比较，弱能量位置按比例衰减。
    期望输入 S 为 [B,1,F,T]（幅度或 log 幅度），返回同形状。
    """
    med = S.median(dim=-1, keepdim=True).values  # [B,1,F,1]
    thr = med * (1.0 + ratio)
    mask = (S >= thr).float()
    S2 = S * mask + (1.0 - mask) * (0.7 * S)
    return S2


def duda_denoise_audio(S: torch.Tensor, args):
    """
    通用音频谱去噪：谱门控 + 退火时/频平滑。
    自动适配输入形状（[B,1,F,T] or [B,F,T] or [B,T,F]），输出保持原形状。
    """
    S4, back = _to_b1ft(S)

    gate  = float(getattr(args, 'duda_a_gate', 0.25))
    steps = int(getattr(args, 'duda_a_steps', 2))
    kt    = int(getattr(args, 'duda_a_kt', 3))
    kf    = int(getattr(args, 'duda_a_kf', 1))

    Y = _spectral_gate(S4, ratio=gate)
    for _ in range(steps):
        Y = _conv1d_along_time(Y, kt)
        if kf > 1:
            Y = _conv1d_along_freq(Y, kf)

    return back(Y)


def duda_apply_for_anchor(a_in: torch.Tensor, v_in: torch.Tensor, which: str, args):
    """
    which in {'audio','video','none'}.
    仅对被污染模态做去腐败。音频/视觉各自保持输入 shape 不变。
    """
    if not bool(getattr(args, 'duda_enable', True)):
        return a_in, v_in
    if which == 'audio' and a_in is not None:
        return duda_denoise_audio(a_in, args), v_in
    if which == 'video' and v_in is not None:
        return a_in, duda_denoise_visual(v_in, args)  # 视觉分支保持你原来的实现
    return a_in, v_in


# ===== Diffusion-guided (no-pretrain) online denoising utils =====
@torch.no_grad()
def _tv_step(x, lam=0.0005, iters=1):
    # 极轻量 total-variation proximal-like 步；x: (B,C,H,W) or (B,T,F)
    for _ in range(iters):
        dx = torch.zeros_like(x)
        dx[..., 1:, :] += x[..., 1:, :] - x[..., :-1, :]
        dx[..., :-1, :] -= x[..., 1:, :] - x[..., :-1, :]
        dx[..., :, 1:] += x[..., :, 1:] - x[..., :, :-1]
        dx[..., :, :-1] -= x[..., :, 1:] - x[..., :, :-1]
        x = x - lam * dx
    return x

@torch.no_grad()
def _annealed_gaussian_smoothing(x, sigmas=(1.0, 0.5), k=3):
    # 退火多视图自一致：多次加噪-平滑-平均；避免过度模糊
    out = 0.0
    cnt = 0
    for s in sigmas:
        if s <= 0:
            out = out + x; cnt += 1;
            continue
        for _ in range(k):
            n = torch.randn_like(x) * s
            x_n = x + n
            # 3x3均值核近似高斯（通道独立），在频谱/图像上都适用
            x_s = torch.nn.functional.avg_pool2d(
                x_n.unsqueeze(1) if x_n.dim()==3 else x_n,
                kernel_size=3, stride=1, padding=1
            )
            if x.dim()==3:  # (B,T,F) -> 去掉伪通道
                x_s = x_s.squeeze(1)
            out = out + x_s
            cnt += 1
    return out / max(cnt, 1)

@torch.no_grad()
def _fft_highfreq_suppress(img, keep_frac=0.6):
    # 图像频域高频抑制（H×W）；img: (B,C,H,W) in [-∞,∞] 已标准化空间
    B, C, H, W = img.shape
    f = torch.fft.fft2(img, norm="ortho")
    f = torch.fft.fftshift(f)

    cy, cx = H // 2, W // 2
    ry, rx = int(cy * keep_frac), int(cx * keep_frac)
    mask = torch.zeros((H, W), device=img.device, dtype=torch.bool)
    mask[cy-ry:cy+ry, cx-rx:cx+rx] = True
    mask = mask.view(1,1,H,W)
    f = f * mask

    f = torch.fft.ifftshift(f)
    rec = torch.fft.ifft2(f, norm="ortho").real
    return rec

@torch.no_grad()
def denoise_audio_spectrogram(a_tensor, args):
    """
    a_tensor: [B, 1, T, F] 或 [B, T, F] 皆可（自动兼容）
    步骤：软谱门 + 退火平滑 + 极轻量TV
    """
    if a_tensor is None:
        return None
    x = a_tensor
    if x.dim()==3:
        x = x.unsqueeze(1)          # -> [B,1,T,F]
    # 软谱门：按频轴和时轴的中位数估计“噪声地板”
    med_f = x.median(dim=-1, keepdim=True).values
    med_t = x.median(dim=-2, keepdim=True).values
    noise_floor = torch.minimum(med_f, med_t)
    alpha = float(getattr(args, "dg_audio_gate", 1.0))
    x = torch.maximum(x - alpha * noise_floor, torch.zeros_like(x))

    # 退火平滑
    x = _annealed_gaussian_smoothing(x.squeeze(1), sigmas=(0.8, 0.4), k=2).unsqueeze(1)
    # 极轻量 TV 步
    x = _tv_step(x, lam=0.0005, iters=1)
    return x if a_tensor.dim()==4 else x.squeeze(1)

@torch.no_grad()
def denoise_visual_tensor(v_tensor, args):
    """
    v_tensor: [B, 3, H, W]
    步骤：FFT 高频抑制 → 退火平滑 → TV 微调
    """
    if v_tensor is None:
        return None
    x = v_tensor
    keep = float(getattr(args, "dg_visual_keep_frac", 0.65))
    x = _fft_highfreq_suppress(x, keep_frac=keep)
    x = _annealed_gaussian_smoothing(x, sigmas=(1.0, 0.5), k=2)
    x = _tv_step(x, lam=0.0003, iters=1)
    return x

@torch.no_grad()
def diffusion_guided_denoise(a_in, v_in, corrupted: str, args):
    """
    根据一次性判定的 corrupted 模态选择性降噪。
    corrupted: 'audio' / 'video' / 'none'
    """
    if not bool(getattr(args, "dg_enable", False)):
        return a_in, v_in
    if corrupted == "audio":
        a_out = denoise_audio_spectrogram(a_in, args)
        return a_out if a_out is not None else a_in, v_in
    elif corrupted == "video":
        v_out = denoise_visual_tensor(v_in, args)
        return a_in, v_out if v_out is not None else v_in
    else:
        return a_in, v_in

# =========================
# 常量 & 名称匹配工具
# =========================
AUDIO_QKV_NAMES = ["blocks_u.0.attn.qa", "blocks_u.0.attn.ka", "blocks_u.0.attn.va", "blocks_u.0.attn.qv", "blocks_u.0.attn.kv", "blocks_u.0.attn.vv"]
VISUAL_QKV_NAMES = ["blocks_u.0.attn.qv", "blocks_u.0.attn.kv", "blocks_u.0.attn.vv", "blocks_u.0.attn.qa", "blocks_u.0.attn.ka", "blocks_u.0.attn.va"]

def _strip_module_prefix(name: str) -> str:
    return name[7:] if name.startswith("module.") else name

def _name_in_list(nm: str, targets: list) -> bool:
    nm2 = _strip_module_prefix(nm)
    return any(nm2.endswith(t) for t in targets)

def _count_trainables(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n, n / 1e6

# =========================
# 冻/解冻：QKV
# =========================
def _freeze_all_qkv(model: nn.Module):
    for nm, m in model.named_modules():
        if _name_in_list(nm, AUDIO_QKV_NAMES + VISUAL_QKV_NAMES):
            for p in m.parameters():
                p.requires_grad = False

def _unfreeze_modality_qkv(model: nn.Module, modality: str):
    targets = AUDIO_QKV_NAMES if modality == "audio" else VISUAL_QKV_NAMES
    for nm, m in model.named_modules():
        if _name_in_list(nm, targets):
            for p in m.parameters():
                p.requires_grad = True
            print(f"[FT] unfreeze QKV: {nm}")

# =========================
# 冻/解冻：Norm 层
# =========================
def _iter_u_block_norms(model):
    core = model.module if hasattr(model, "module") else model
    for blk in core.blocks_u:
        for n, m in blk.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                yield n, m

def _iter_single_branch_norms(model, branch: str):  # branch in {"a","v"}
    core = model.module if hasattr(model, "module") else model
    blocks = core.blocks_a if branch == "a" else core.blocks_v
    for blk in blocks:
        for n, m in blk.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                yield n, m

def _freeze_all_norms(model):
    # U-block
    for n, m in _iter_u_block_norms(model):
        for pn, p in m.named_parameters(recurse=False):
            if pn in ("weight", "bias"):
                p.requires_grad = False
    # 单模态两支
    for branch in ("a", "v"):
        for n, m in _iter_single_branch_norms(model, branch):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias"):
                    p.requires_grad = False

def _unfreeze_u_block_norms(model):
    for n, m in _iter_u_block_norms(model):
        for pn, p in m.named_parameters(recurse=False):
            if pn in ("weight", "bias"):
                p.requires_grad = True
        if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            m.track_running_stats = False
            if hasattr(m, "running_mean"): m.running_mean = None
            if hasattr(m, "running_var"):  m.running_var  = None
    print("[FT] unfreeze all norms in U-block")

def _unfreeze_single_branch_norms(model, branch: str):
    # branch: "a" 表示 blocks_a；"v" 表示 blocks_v
    for n, m in _iter_single_branch_norms(model, branch):
        for pn, p in m.named_parameters(recurse=False):
            if pn in ("weight", "bias"):
                p.requires_grad = True
        if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            m.track_running_stats = False
            if hasattr(m, "running_mean"): m.running_mean = None
            if hasattr(m, "running_var"):  m.running_var  = None
    print(f"[FT] unfreeze norms in blocks_{branch}")

# =========================
# MLA 权重
# =========================
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

# =========================
# JS 判定（一次性校准/懒校准复用）
# =========================
def _js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (p.add(eps).log() - m.add(eps).log()), dim=1)
    kl_qm = torch.sum(q * (q.add(eps).log() - m.add(eps).log()), dim=1)
    return 0.5 * (kl_pm + kl_qm)

def decide_corrupted_modality_js(p_f: torch.Tensor,
                                 p_a: torch.Tensor,
                                 p_v: torch.Tensor,
                                 args) -> str:
    """
    基于 JS(p_f||p_a)、JS(p_f||p_v) 的单次判定，返回 'audio' / 'video' / 'none'
    """
    eps = 1e-6
    d_a = _js_div(p_f, p_a, eps=eps).mean().item()
    d_v = _js_div(p_f, p_v, eps=eps).mean().item()
    delta = d_v - d_a  # >0 → visual 更远；<0 → audio 更远

    # 融合过于不确定时不判
    # C = p_f.size(1); logC = math.log(C + 1e-12)
    # Hf = (-(p_f * p_f.add(eps).log()).sum(dim=1)).mean().item() / logC
    # if Hf > float(getattr(args, "decide_max_fused_entropy", 0.95)):
    #     return "none"

    margin = float(getattr(args, "decide_margin", 0.0005))
    if   delta >  margin: return "video"
    elif delta < -margin: return "audio"
    else:                 return "none"

# =========================
# （可选）正式校准：用前K个样本
# =========================
@torch.no_grad()
def calibrate_corruption(model: nn.Module, loader, args, max_samples: int = 100):
    """
    用前 max_samples 个样本做一次性判定并**固定**微调集合；设置 args._corr_fixed
    你也可以不调用此函数，forward_and_adapt 会在首次 batch 上做“懒校准”
    """
    core = model.module if hasattr(model, "module") else model
    device = next(core.parameters()).device
    n_seen = 0
    acc_pf = acc_pa = acc_pv = None

    for a_input, v_input, _labels in loader:
        a_input = a_input.to(device, non_blocking=True)
        v_input = v_input.to(device, non_blocking=True)

        # fused
        out_f = core.forward_eval(a=a_input, v=v_input, mode=args.testmode, args=args,
                                  stage='test', labels=None, ret_av_feat=False, output_embedding=False)
        logits_f = out_f[0] if isinstance(out_f, (list, tuple)) else out_f
        p_f = torch.softmax(logits_f.float(), dim=1).clamp_min(1e-6)

        # unimodal
        logits_a, _ = core.forward_eval(a=a_input, v=None,  mode='audio',  args=args, stage='test')
        logits_v, _ = core.forward_eval(a=None,  v=v_input, mode='visual', args=args, stage='test')
        p_a = torch.softmax(logits_a.float(), dim=1).clamp_min(1e-6)
        p_v = torch.softmax(logits_v.float(), dim=1).clamp_min(1e-6)

        b = p_f.size(0)
        if acc_pf is None:
            acc_pf = p_f.mean(dim=0)
            acc_pa = p_a.mean(dim=0)
            acc_pv = p_v.mean(dim=0)
        else:
            acc_pf = (acc_pf * n_seen + p_f.sum(dim=0)) / (n_seen + b)
            acc_pa = (acc_pa * n_seen + p_a.sum(dim=0)) / (n_seen + b)
            acc_pv = (acc_pv * n_seen + p_v.sum(dim=0)) / (n_seen + b)
        n_seen += b
        if n_seen >= max_samples:
            break

    if n_seen == 0:
        args._corr_fixed = "none"
        print("[CALIB] no samples; set corruption=none")
        return "none"

    pf = acc_pf.unsqueeze(0); pa = acc_pa.unsqueeze(0); pv = acc_pv.unsqueeze(0)
    corr = decide_corrupted_modality_js(pf, pa, pv, args)
    args._corr_fixed = corr
    print(f"[CALIB] corruption decided: {corr} (seen {n_seen} samples)")

    # 一次到位切换微调集合
    _freeze_all_qkv(model); _freeze_all_norms(model)
    if corr == "audio":
        _unfreeze_modality_qkv(model, "audio")
        _unfreeze_u_block_norms(model)
        _unfreeze_single_branch_norms(model, "a")
        print("[CALIB] FT-audio QKV + U-block norms + blocks_a norms")
    elif corr == "video":
        _unfreeze_modality_qkv(model, "video")
        _unfreeze_u_block_norms(model)
        _unfreeze_single_branch_norms(model, "v")
        print("[CALIB] FT-visual QKV + U-block norms + blocks_v norms")
    else:
        print("[CALIB] FT-none (all frozen)")

    args._last_corr = corr
    return corr

# =========================
# 其余（Cache / 工具）
# =========================
class Cache:
    def __init__(self, size) -> None:
        self.features = []
        self.sample_attns = []
        self.labels = []
        self.size = size
    def update(self):
        if len(self.features) > self.size:      self.features = self.features[1:]
        if len(self.sample_attns) > self.size:  self.sample_attns = self.sample_attns[1:]
        if len(self.labels) > self.size:        self.labels = self.labels[1:]
    def sample_feature(self):
        idx = random.randint(0, len(self.features) - 1)
        return self.features[idx]

def kl_div(p_mean, p_std, q_mean, q_std):
    return torch.log(q_std / p_std) + (p_std ** 2 + (p_mean - q_mean) ** 2) / (2 * q_std ** 2) - 0.5

# =========================
# READ wrapper
# =========================
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
                loss_tuple = (torch.tensor(0.0),)*5
            return outputs, loss_tuple

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# =========================
# 主：forward_and_adapt（一次性“懒校准” + 固定微调集合）
# =========================
@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    """
    ABPEM (with REM-style RE) + PoE/MLA consistency + Adaptive Rank Weighting.
    先校准（若尚未校准则用当前 batch 懒校准），固定微调集合；若判定 none 则直接 test-stage 返回。
    """

    core = model.module if hasattr(model, "module") else model
    eps = 1e-6
    # ====== A) 在取 anchors 前，根据已锁定的腐败模态做一次输入级降噪（无梯度） ======
    with torch.no_grad():
        corr_fixed = getattr(args, "_corr_fixed", None)
        a_in, v_in = x[0], x[1]
        if corr_fixed is not None:
            a_in, v_in = diffusion_guided_denoise(a_in, v_in, corr_fixed, args)
            x = (a_in, v_in)
    # ---------- (2) 先用“原始”单模态 anchor 做一次判定 ----------
    a_raw, v_raw = x[0], x[1]
    logits_a_raw, _ = model.module.forward_eval(a=a_raw, v=None,  mode='audio',  args=args, stage='test')
    logits_v_raw, _ = model.module.forward_eval(a=None,  v=v_raw, mode='visual', args=args, stage='test')

    # 用 KL/entropy 判定哪侧更像被污染（你已有 decide_* 可替代）
    eps = 1e-6
    pa = torch.softmax(logits_a_raw.float(), dim=1).clamp_min(eps)
    pv = torch.softmax(logits_v_raw.float(), dim=1).clamp_min(eps)
    C  = pa.size(1); logC = math.log(C + 1e-12)
    Ha = -(pa * pa.log()).sum(dim=1).mean().item() / logC
    Hv = -(pv * pv.log()).sum(dim=1).mean().item() / logC
    corrupted = 'audio' if (Ha - Hv) > float(getattr(args,'tau_ent_hi',0.10)) else \
                'video' if (Hv - Ha) > float(getattr(args,'tau_ent_hi',0.10)) else 'none'

    # ---------- (2.1) 对“被判定的那一侧”做 DUDA，然后再提 anchor ----------
    a_anchor_in, v_anchor_in = duda_apply_for_anchor(a_raw, v_raw, which=corrupted, args=args)
    logits_a_anchor, _ = model.module.forward_eval(a=a_anchor_in, v=None,  mode='audio',  args=args, stage='test')
    logits_v_anchor, _ = model.module.forward_eval(a=None,  v=v_anchor_in, mode='visual', args=args, stage='test')


    # ---------- (1) 若尚未校准：用当前 batch 懒校准一次 ----------
    if not hasattr(args, "_corr_fixed"):
        with torch.no_grad():
            out_f = core.forward_eval(
                a=x[0], v=x[1], mode=args.testmode, args=args,
                stage='test', labels=labels, ret_av_feat=False, output_embedding=False
            )
            logits_f = out_f[0] if isinstance(out_f, (list, tuple)) else out_f

            p_f = torch.softmax(logits_f.float(), dim=1).clamp_min(1e-6)
            p_a = torch.softmax(logits_a_anchor.float(), dim=1).clamp_min(1e-6)
            p_v = torch.softmax(logits_v_anchor.float(), dim=1).clamp_min(1e-6)

            corr = decide_corrupted_modality_js(p_f, p_a, p_v, args)
            args._corr_fixed = corr
            print(f"[LAZY-CALIB] corruption decided: {corr}")

            _freeze_all_qkv(model); _freeze_all_norms(model)
            if corr == "audio":
                _unfreeze_modality_qkv(model, "audio")
                _unfreeze_u_block_norms(model)
                _unfreeze_single_branch_norms(model, "a")
            elif corr == "video":
                _unfreeze_modality_qkv(model, "video")
                _unfreeze_u_block_norms(model)
                _unfreeze_single_branch_norms(model, "v")
            else:
                pass  # 全冻

            # ====== B) 懒校准后，对当前批做同样降噪以保持一致 ======
            a_in2, v_in2 = diffusion_guided_denoise(x[0], x[1], corr, args)
            x = (a_in2, v_in2)

            # 切换后建议再无梯度重算 anchors（更一致）
            logits_a_anchor, _ = core.forward_eval(a=x[0], v=None,  mode='audio',  args=args, stage='test')
            logits_v_anchor, _ = core.forward_eval(a=None,  v=x[1], mode='visual', args=args, stage='test')

    corrupted = getattr(args, "_corr_fixed", "none")

    # ---------- (2) 若 none：不适配，直接 test-stage 返回 ----------
    if corrupted == "none":
        with torch.no_grad():
            out_test = core.forward_eval(
                a=x[0], v=x[1], mode=args.testmode, args=args,
                stage='test', labels=labels, ret_av_feat=False, output_embedding=False
            )
        out_test = out_test[0] if isinstance(out_test, (list, tuple)) else out_test
        zero = torch.tensor(0.0, device=out_test.device)
        return (out_test, out_test), (zero.item(), zero.item(), zero.item(), zero.item(), zero.item())

    # ---------- (3) 正式适配（带图） ----------
    with autocast():
        out_pack = core.forward_eval(
            a=x[0], v=x[1],
            mode=args.testmode, args=args,
            stage='adapt', labels=labels,
            ret_av_feat=True, output_embedding=True
        )
    if isinstance(out_pack, (list, tuple)) and len(out_pack) == 3:
        outputs, attn_raw, cached_av_feat = out_pack
    else:
        outputs, attn_raw = out_pack
        cached_av_feat = None

    # ---------- (4) MLA 权重 ----------
    q_a, q_v = compute_modality_weights_mla(
        logits_a_anchor, logits_v_anchor, tau=getattr(args, 'mla_tau', 1.0)
    )
    gap_val = float((q_a - q_v).abs().mean().item())

    # ---------- (5) Core losses ----------
    pred = outputs.softmax(dim=-1)                                  # [B, C]
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)         # [B, C]
    C = p_sorted.size(1)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)

    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam1 = getattr(args, "re_lambda1", 0.0)
    lam_eff = max(1e-6, float(lam0 + lam1 * gap_val))
    w = torch.exp(-lam_eff * rank); w = w / (w.sum() + 1e-12)       # [C]

    ent_sorted = -p_sorted * torch.log(p_sorted + 1e-6)
    loss_ra = (ent_sorted * w.unsqueeze(0)).sum(dim=-1).mean()

    if getattr(args, "fix_loss_bal", False):
        p_sum = outputs.softmax(dim=-1).mean(dim=-2)
        loss_bal = - (p_sum * p_sum.log()).sum()
    else:
        p_sum = outputs.softmax(dim=-1).sum(dim=-2)
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()

    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = torch.nn.functional.cross_entropy(outputs, pseudo)

    # ---------- (6) 一致性（PoE；你也可改回 MLA 双 KL） ----------
    p_f = pred
    p_a = torch.softmax(logits_a_anchor, dim=1).clamp_min(eps)
    p_v = torch.softmax(logits_v_anchor, dim=1).clamp_min(eps)
    with torch.no_grad():
        logC = math.log(p_f.size(1) + 1e-12)
        Ha = -(p_a * p_a.log()).sum(dim=1) / logC
        Hv = -(p_v * p_v.log()).sum(dim=1) / logC
        ra = 1.0 / (Ha + 1e-6)
        rv = 1.0 / (Hv + 1e-6)
        gamma = (ra / (ra + rv)).unsqueeze(1)
    log_t = gamma * p_a.log() + (1.0 - gamma) * p_v.log()
    t = torch.softmax(log_t, dim=1).clamp_min(eps)
    loss_cons = torch.sum(p_f * (p_f.add(eps).log() - t.log()), dim=1).mean()

    # ---------- (7) Total & Step ----------
    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
            + loss_ra * getattr(args, "ent_weight", 0.0)
            - loss_bal * getattr(args, "bal_weight", 0.0)
            + loss_cons * getattr(args, "cons_weight", 0.2))

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0
    )
    scaler.step(optimizer)
    scaler.update()

    # ---------- (8) 报告前向（test-stage） ----------
    with torch.no_grad():
        with autocast():
            out2 = core.forward_eval(
                a=x[0], v=x[1], mode=args.testmode, args=args,
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

# =========================
# 参数收集（保持你原接口）
# =========================
requires_grad_list = [
    'module.blocks_u.0.attn.qa',
    'module.blocks_u.0.attn.ka',
    'module.blocks_u.0.attn.va',
    'module.blocks_u.0.attn.qv',
    'module.blocks_u.0.attn.kv',
    'module.blocks_u.0.attn.vv'
]

def collect_params(model):
    """
    仍然收集 QKV（不看 requires_grad），这样切换时只靠 requires_grad 控制是否更新，
    optimizer 的 param group 可保持静态。
    """
    params_fusion_qkv = []
    names_fusion_qkv = []
    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")
    # 也把所有 LN/GN/BN 的 weight/bias 放进去（是否更新由 requires_grad 决定）
    core = model.module if hasattr(model, "module") else model
    for mod in core.modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            for pn, p in mod.named_parameters(recurse=False):
                if pn in ("weight", "bias"):
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{type(mod).__name__}.{pn}")
    print(f"[INIT] Collected candidate FT params: {len(params_fusion_qkv)} tensors.")
    return params_fusion_qkv, names_fusion_qkv

# =========================
# 复制/加载优化器
# =========================
def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

# =========================
# 配置：初始全冻结（等待校准）
# =========================
def configure_model(model):
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    print("[INIT] all QKV & all norms are frozen; wait for corruption decision.")

    return model







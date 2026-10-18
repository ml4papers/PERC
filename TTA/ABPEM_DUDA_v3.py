# ABPEM.py  ———— DUDA集成版（drop-in replacement）
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
import torch.nn.functional as F

# ===============================
# DUDA: Diffusion-inspired, score-free, test-time denoiser (3 knobs)
# ===============================
# ========== Corruption mapping ==========
CORR_CATEGORY = {
    # ---- video noise ----
    "gaussian_noise": "noise",
    "shot_noise": "noise",
    "impulse_noise": "noise",

    # ---- video blur/digital ----
    "defocus_blur": "digital",
    "glass_blur": "digital",
    "motion_blur": "digital",
    "zoom_blur": "digital",
    "elastic_transform": "digital",
    "pixelate": "digital",
    "jpeg_compression": "digital",

    # ---- video weather/contrast ----
    "snow": "weather",
    "frost": "weather",
    "fog": "weather",
    "brightness": "weather",
    "contrast": "weather",

    # ---- audio noise ----
    "traffic": "noise",
    "crowd": "noise",
    "wind": "noise",

    # ---- audio weather ----
    "rain": "weather",
    "thunder": "weather",
}

def gaussian_blur(x, kernel_size=5, sigma=1.0):
    if x is None: return None
    coords = torch.arange(kernel_size, device=x.device) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    g2d = g[:, None] * g[None, :]
    g2d = g2d.expand(x.size(1), 1, kernel_size, kernel_size)
    return F.conv2d(x, g2d, padding=kernel_size//2, groups=x.size(1))

def sobel_filter(x):
    if x is None: return None
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device)
    sobel_x = sobel_x.expand(x.size(1),1,3,3)
    sobel_y = sobel_y.expand(x.size(1),1,3,3)
    grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
    grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    return x + grad

def apply_defilter(x, corr_name):
    if x is None or corr_name is None:
        return x
    corr_type = CORR_CATEGORY.get(corr_name, "none")
    if corr_type in ["noise", "weather"]:
        return gaussian_blur(x)
    elif corr_type == "digital":
        return sobel_filter(x)
    else:
        return x


def _lerp(a, b, t):  # linear interpolate
    return a + (b - a) * float(max(0.0, min(1.0, t)))

@torch.no_grad()
def _entropy_norm(probs: torch.Tensor, eps=1e-6):
    """ 归一化熵 in [0,1] """
    C = probs.size(1)
    H  = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1)
    Hn = H / math.log(C + 1e-12)
    return Hn  # [B]

@torch.no_grad()
def _kl_pq(p, q, eps=1e-6):
    return torch.sum(p * (p.add(eps).log() - q.add(eps).log()), dim=1)

# ---------- 简单统计：音频谱/视觉特征的噪声与模糊指示 ----------
@torch.no_grad()
def _stats_audio_spec(S: torch.Tensor):
    """
    S: [B,1,F,T] 线性幅度或对数幅度谱
    返回：SFM（谱平坦度∈[0,1], 噪声指示）、TR（瞬态率）、LFC（低频占比）、blur_like（是否偏模糊）
    """
    B, C, F, T = S.shape
    eps = 1e-6
    X = S.float().abs() + eps
    geo = torch.exp(torch.log(X + eps).mean(dim=(2,3)))          # [B,1]
    ari = (X.mean(dim=(2,3)) + eps)                              # [B,1]
    sfm = (geo / ari).clamp(0, 1).squeeze(1)                     # [B]
    f_cut = max(4, F // 6)
    lfc = (X[:, :, :f_cut, :].sum(dim=(2,3)) / (X.sum(dim=(2,3)) + eps)).squeeze(1)  # [B]
    dt = (X[:, :, :, 1:] - X[:, :, :, :-1]).abs().mean(dim=(2,3)) / (X.mean(dim=(2,3)) + eps)  # [B,1]
    tr = dt.squeeze(1).clamp(0,1)
    hf = (X[:, :, F//2:, :].sum(dim=(2,3)) / (X.sum(dim=(2,3)) + eps)).squeeze(1)  # 高频占比
    blur_like = (hf < 0.30) & (lfc > 0.45)
    return sfm, tr, lfc, blur_like

@torch.no_grad()
def _stats_visual_feat(V: torch.Tensor):
    """
    V: [B, T, D] 或 [B, N, D] 的视觉token，或 [B,C,H,W] 的浅层特征/图像
    返回：hf_ratio, edge, blur_like
    """
    if V.dim() == 4:  # [B,C,H,W]
        X = V.float()
    else:             # [B,N,D] => 近似 reshape 成 [B, D, sqrtN, sqrtN]
        B, N, D = V.shape
        S = int(math.sqrt(max(1, N)))
        if S*S != N:
            N2 = S*S
            V = V[:, :N2, :]
        X = V.transpose(1,2).reshape(B, D, S, S).contiguous()

    B, C, H, W = X.shape
    eps = 1e-6
    gx = F.conv2d(X, weight=_sobel_kx(X.device, C), padding=1, groups=C)
    gy = F.conv2d(X, weight=_sobel_ky(X.device, C), padding=1, groups=C)
    edge = (gx.abs() + gy.abs()).mean(dim=(1,2,3)) + eps         # [B]
    lap = F.conv2d(X, weight=_laplace_k(X.device, C), padding=1, groups=C).abs()
    hf_ratio = (lap.mean(dim=(1,2,3)) / (X.abs().mean(dim=(1,2,3)) + eps)).clamp(0, 1)  # [B]
    blur_like = (hf_ratio < 0.25) & (edge < 0.10)
    return hf_ratio, edge, blur_like

_sobel_cache = {}
def _sobel_kx(dev, ch):
    key=("kx",ch,dev)
    if key in _sobel_cache: return _sobel_cache[key]
    k = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32, device=dev)
    k = k.repeat(ch,1,1,1)
    _sobel_cache[key]=k
    return k
def _sobel_ky(dev, ch):
    key=("ky",ch,dev)
    if key in _sobel_cache: return _sobel_cache[key]
    k = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32, device=dev)
    k = k.repeat(ch,1,1,1)
    _sobel_cache[key]=k
    return k
def _laplace_k(dev, ch):
    key=("lap",ch,dev)
    if key in _sobel_cache: return _sobel_cache[key]
    k = torch.tensor([[[0,-1,0],[-1,4,-1],[0,-1,0]]], dtype=torch.float32, device=dev)
    k = k.repeat(ch,1,1,1)
    _sobel_cache[key]=k
    return k

@torch.no_grad()
def duda_denoise_audio_anchor(S: torch.Tensor, args):
    """
    S: [B,1,F,T] 或 [B,F,T] / [B,T,F]
    返回：和输入同布局的 S_hat
    """
    if S is None:
        return None
    X4, desc = _spec_to_4d(S, args)  # -> [B,1,F,T]

    eps = 1e-6
    B, C, F, T = X4.shape
    strength = float(getattr(args, "duda_strength", 0.5))
    protect  = float(getattr(args, "duda_protect",  0.6))
    smooth   = float(getattr(args, "duda_smooth",   0.5))

    # 统计与自适应参数
    sfm, tr, lfc, blur_like = _stats_audio_spec(X4)  # 注意：这里传入 [B,1,F,T]
    gate_base = _lerp(0.35, 0.70, strength) * (0.6 + 0.4*sfm.mean().item())
    tv_w      = _lerp(1e-3, 6e-3, strength) * (0.5 + 0.5*sfm.mean().item())
    gamma0    = _lerp(0.90, 0.97, smooth)
    gamma1    = _lerp(0.70, 0.92, smooth)

    if blur_like.float().mean().item() > 0.5:
        gate_base *= 0.85
        tv_w      *= 0.8
        protect    = max(protect, 0.7)
        gamma0     = min(gamma0, 0.95)

    # 软门 + 退火平滑 + 轻TV
    mag = X4.abs()
    thr = gate_base * (mag.mean(dim=(2,3), keepdim=True) + eps)
    mask = (mag > thr).float()
    onset_alpha = 1.0 - 0.5 * tr.clamp(0,1).mean().item() * protect
    Y = mask * X4 + (1-mask) * (onset_alpha * X4)

    Yt = _exp_smooth_time(Y, gamma0)
    Yt = _exp_smooth_time(Yt.flip(-1), gamma1).flip(-1)
    Yf = _exp_smooth_freq(Y, gamma0)
    Yf = _exp_smooth_freq(Yf.flip(-2), gamma1).flip(-2)
    Y  = 0.5*Yt + 0.5*Yf

    if tv_w > 0:
        Y = Y - tv_w * _laplace2d(Y)

    # 还原到输入的布局
    return _spec_from_4d(Y, desc)


@torch.no_grad()
def duda_denoise_visual_feat(V: torch.Tensor, args):
    """
    V: 融合前视觉特征或浅层图像 [B,C,H,W] 或 [B,N,D]
    返回 V_hat（特征空间轻度去噪/去振铃）
    """
    if V is None: return None
    strength = float(getattr(args, "duda_strength", 0.5))
    protect  = float(getattr(args, "duda_protect",  0.6))
    smooth   = float(getattr(args, "duda_smooth",   0.5))

    if V.dim() == 4:
        X = V.float()
    else:
        B, N, D = V.shape
        S = int(math.sqrt(max(1, N)))
        if S*S != N:
            V = V[:, :S*S, :]
        X = V.transpose(1,2).reshape(B, D, S, S).contiguous()

    hf_ratio, edge, blur_like = _stats_visual_feat(X)
    tv_w   = _lerp(5e-4, 3e-3, strength) * (0.5 + 0.5*(1.0 - hf_ratio.mean().item()))
    gamma0 = _lerp(0.90, 0.97, smooth)
    gamma1 = _lerp(0.70, 0.92, smooth)

    if blur_like.float().mean().item() > 0.5:
        tv_w   *= 0.7
        gamma0  = min(gamma0, 0.95)
        protect = max(protect, 0.7)

    Y = X
    Yt = _exp_smooth_time_2d(Y, gamma0, dim=-1)
    Yt = _exp_smooth_time_2d(Yt.flip(-1), gamma1, dim=-1).flip(-1)
    Yf = _exp_smooth_time_2d(Y, gamma0, dim=-2)
    Yf = _exp_smooth_time_2d(Yf.flip(-2), gamma1, dim=-2).flip(-2)
    Y  = 0.5*Yt + 0.5*Yf

    if tv_w > 0:
        Y = Y - tv_w * _laplace2d(Y)

    if V.dim() != 4:
        B, D, S, S2 = Y.shape
        Y = Y.reshape(B, D, S*S2).transpose(1,2).contiguous()

    return Y

def _exp_smooth_time(S, gamma):
    B,C,F,T = S.shape
    Y = S.clone()
    for t in range(1, T):
        Y[..., t] = gamma*Y[..., t-1] + (1-gamma)*S[..., t]
    return Y

def _exp_smooth_freq(S, gamma):
    B,C,F,T = S.shape
    Y = S.clone()
    for f in range(1, F):
        Y[:, :, f, :] = gamma*Y[:, :, f-1, :] + (1-gamma)*S[:, :, f, :]
    return Y

def _exp_smooth_time_2d(X, gamma, dim=-1):
    Y = X.clone()
    T = X.size(dim)
    slc_prev = [slice(None)]*X.dim()
    slc_this = [slice(None)]*X.dim()
    for t in range(1, T):
        slc_prev[dim] = t-1
        slc_this[dim] = t
        Y[tuple(slc_this)] = gamma*Y[tuple(slc_prev)] + (1-gamma)*X[tuple(slc_this)]
    return Y

def _laplace2d(X):
    B,C,H,W = X.shape
    k = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=X.dtype, device=X.device)
    k = k.repeat(C,1,1,1)
    return F.conv2d(X, k, padding=1, groups=C)

@torch.no_grad()
def duda_apply_for_anchor(a_raw, v_raw, which: str, args, model=None, fused_logits=None):
    """
    a_raw: [B,1,F,T] 频谱（你的 dataloader 已经这样给）
    v_raw: 视觉输入或中间特征（根据你的接口可传 None）
    which: 'audio' / 'video' / 'none'
    fused_logits: 可选，用于自监控（若提供，则计算 DUDA 前后与 fused 的 KL；不提供则只看熵）
    返回: (a_in, v_in) 作为 forward_eval 的输入（仅对坏模态替换为去噪结果）
    """
    if which == 'audio':
        a_dn = duda_denoise_audio_anchor(a_raw, args)
        if model is not None:
            la0, _ = model.module.forward_eval(a=a_raw, v=None, mode='audio', args=args, stage='test')
            la1, _ = model.module.forward_eval(a=a_dn,  v=None, mode='audio', args=args, stage='test')
            pa0 = torch.softmax(la0, dim=1); pa1 = torch.softmax(la1, dim=1)
            H0 = _entropy_norm(pa0).mean();   H1 = _entropy_norm(pa1).mean()
            good = (H1 <= H0 + 1e-3)
            if fused_logits is not None:
                pf = torch.softmax(fused_logits, dim=1)
                kl0 = _kl_pq(pa0, pf).mean(); kl1 = _kl_pq(pa1, pf).mean()
                good = good & (kl1 <= kl0 + 1e-3)
            if not bool(good.item()):
                a_dn = a_raw
        return a_dn, v_raw

    if which == 'video':
        v_dn = duda_denoise_visual_feat(v_raw, args)
        if model is not None:
            lv0, _ = model.module.forward_eval(a=None, v=v_raw, mode='visual', args=args, stage='test')
            lv1, _ = model.module.forward_eval(a=None, v=v_dn,  mode='visual', args=args, stage='test')
            pv0 = torch.softmax(lv0, dim=1); pv1 = torch.softmax(lv1, dim=1)
            H0 = _entropy_norm(pv0).mean();  H1 = _entropy_norm(pv1).mean()
            good = (H1 <= H0 + 1e-3)
            if fused_logits is not None:
                pf = torch.softmax(fused_logits, dim=1)
                kl0 = _kl_pq(pv0, pf).mean(); kl1 = _kl_pq(pv1, pf).mean()
                good = good & (kl1 <= kl0 + 1e-3)
            if not bool(good.item()):
                v_dn = v_raw
        return a_raw, v_dn

    return a_raw, v_raw

# ===== 可选：简易diffusion-guided预处理（若想在进入模型前就处理原始输入） =====
@torch.no_grad()
def denoise_audio_spectrogram(a_tensor, args):
    """
    a_tensor: [B,1,F,T] 或 [B,F,T] / [B,T,F]
    返回：与输入同布局的张量
    """
    if a_tensor is None:
        return None
    X4, desc = _spec_to_4d(a_tensor, args)  # -> [B,1,F,T]

    # 软谱门：频/时双向中位数估计噪声地板
    med_f = X4.median(dim=-1, keepdim=True).values
    med_t = X4.median(dim=-2, keepdim=True).values
    noise_floor = torch.minimum(med_f, med_t)
    alpha = float(getattr(args, "dg_audio_gate", 1.0))
    X4 = torch.maximum(X4 - alpha * noise_floor, torch.zeros_like(X4))

    # 退火平滑 + 轻 TV
    X4 = _annealed_gaussian_smoothing(X4, sigmas=(0.8, 0.4), k=2)
    X4 = _tv_step(X4, lam=0.0005, iters=1)

    return _spec_from_4d(X4, desc)


@torch.no_grad()
def denoise_visual_tensor(v_tensor, args):
    if v_tensor is None: return None
    x = v_tensor
    keep = float(getattr(args, "dg_visual_keep_frac", 0.65))
    x = _fft_highfreq_suppress(x, keep_frac=keep)
    x = _annealed_gaussian_smoothing(x, sigmas=(1.0, 0.5), k=2)
    x = _tv_step(x, lam=0.0003, iters=1)
    return x

@torch.no_grad()
def _tv_step(x, lam=0.0005, iters=1):
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
    out = 0.0
    cnt = 0
    for s in sigmas:
        if s <= 0:
            out = out + x; cnt += 1
            continue
        for _ in range(k):
            n = torch.randn_like(x) * s
            x_n = x + n
            x_s = torch.nn.functional.avg_pool2d(
                x_n if x_n.dim()==4 else x_n.unsqueeze(1),
                kernel_size=3, stride=1, padding=1
            )
            x_s = x_s if x_n.dim()==4 else x_s.squeeze(1)
            out = out + x_s
            cnt += 1
    return out / max(cnt, 1)

@torch.no_grad()
def _fft_highfreq_suppress(img, keep_frac=0.6):
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
def diffusion_guided_denoise(a_in, v_in, corrupted: str, args):
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


# ==== NEW: robust spec reshape helpers ====
@torch.no_grad()
def _spec_to_4d(S: torch.Tensor, args):
    """
    Accept [B,1,F,T] or [B,F,T] or [B,T,F].
    Return X4: [B,1,F,T] and a descriptor to restore original layout.
    """
    desc = {"was4d": False, "order": "FT"}
    if S.dim() == 4:
        # [B,1,F,T] (or [B,C,F,T] but your pipeline用的是C=1)
        desc["was4d"] = True
        return S, desc

    # 3D -> decide which is F and T
    assert S.dim() == 3, f"expect 3D/4D spec, got {S.shape}"
    B, D1, D2 = S.shape
    # 经验性假设：F(梅尔bins)更小（常见 64/80/128），T 更大（~1k）
    # 如有 args.num_mel_bins 则优先用
    nbins = int(getattr(args, "num_mel_bins", getattr(args, "mel_bins", 128)))
    # 优先匹配包含 nbins 的维度；否则用 min 作为 F
    if D1 == nbins or (D1 < D2 and D1 <= 256):
        # [B,F,T]
        X = S
        desc["order"] = "FT"
    elif D2 == nbins or (D2 < D1 and D2 <= 256):
        # [B,T,F] -> 转 FT
        X = S.transpose(1, 2).contiguous()
        desc["order"] = "TF"
    else:
        # fallback：把较小的一维当 F
        if D1 <= D2:
            X = S
            desc["order"] = "FT"
        else:
            X = S.transpose(1, 2).contiguous()
            desc["order"] = "TF"
    X4 = X.unsqueeze(1)  # [B,1,F,T]
    return X4, desc

@torch.no_grad()
def _spec_from_4d(X4: torch.Tensor, desc):
    """
    Restore back to original layout described by `desc`.
    X4: [B,1,F,T] -> [B,F,T] or [B,T,F] or keep 4D.
    """
    if desc.get("was4d", False):
        return X4
    X = X4.squeeze(1)  # [B,F,T]
    if desc.get("order", "FT") == "FT":
        return X
    else:
        return X.transpose(1, 2).contiguous()  # [B,T,F]


# =========================
# 常量 & 名称匹配工具
# =========================
AUDIO_QKV_NAMES = ["blocks_u.0.attn.qa", "blocks_u.0.attn.ka", "blocks_u.0.attn.va"]
VISUAL_QKV_NAMES = ["blocks_u.0.attn.qv", "blocks_u.0.attn.kv", "blocks_u.0.attn.vv"]

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
    for blk in getattr(core, "blocks_u", []):
        for n, m in blk.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                yield n, m

def _iter_single_branch_norms(model, branch: str):  # branch in {"a","v"}
    core = model.module if hasattr(model, "module") else model
    blocks = getattr(core, "blocks_a", None) if branch == "a" else getattr(core, "blocks_v", None)
    if blocks is None: return
    for blk in blocks:
        for n, m in blk.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                yield n, m

def _freeze_all_norms(model):
    for n, m in _iter_u_block_norms(model):
        for pn, p in m.named_parameters(recurse=False):
            if pn in ("weight", "bias"): p.requires_grad = False
    for branch in ("a", "v"):
        for n, m in _iter_single_branch_norms(model, branch):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias"): p.requires_grad = False

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
    eps = 1e-6
    d_a = _js_div(p_f, p_a, eps=eps).mean().item()
    d_v = _js_div(p_f, p_v, eps=eps).mean().item()
    delta = d_v - d_a  # >0 → visual 更远；<0 → audio 更远
    margin = float(getattr(args, "decide_margin", 0.0005))
    if   delta >  margin: return "video"
    elif delta < -margin: return "audio"
    else:                 return "none"

# =========================
# 正式校准：用前K个样本（可选）
# =========================
@torch.no_grad()
def calibrate_corruption(model: nn.Module, loader, args, max_samples: int = 100):
    core = model.module if hasattr(model, "module") else model
    device = next(core.parameters()).device
    n_seen = 0
    acc_pf = acc_pa = acc_pv = None

    for a_input, v_input, _labels in loader:
        a_input = a_input.to(device, non_blocking=True)
        v_input = v_input.to(device, non_blocking=True)

        out_f = core.forward_eval(a=a_input, v=v_input, mode=args.testmode, args=args,
                                  stage='test', labels=None, ret_av_feat=False, output_embedding=False)
        logits_f = out_f[0] if isinstance(out_f, (list, tuple)) else out_f
        p_f = torch.softmax(logits_f.float(), dim=1).clamp_min(1e-6)

        logits_a, _ = core.forward_eval(a=a_input, v=None,  mode='audio',  args=args, stage='test')
        logits_v, _ = core.forward_eval(a=None,  v=v_input, mode='visual', args=args, stage='test')
        p_a = torch.softmax(logits_a.float(), dim=1).clamp_min(1e-6)
        p_v = torch.softmax(logits_v.float(), dim=1).clamp_min(1e-6)

        b = p_f.size(0)
        if acc_pf is None:
            acc_pf = p_f.mean(dim=0); acc_pa = p_a.mean(dim=0); acc_pv = p_v.mean(dim=0)
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
# 主：forward_and_adapt（一次性“懒校准” + 固定微调集合 + DUDA anchors）
# =========================
@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    """
    ABPEM (with REM-style RE) + PoE/MLA consistency + Adaptive Rank Weighting.
    先校准（若尚未校准则用当前 batch 懒校准），固定微调集合；若判定 none 则直接 test-stage 返回。
    """
    core = model.module if hasattr(model, "module") else model
    eps = 1e-6

    # ====== A) 若已固定腐败模态，先对原始输入做一次轻度预处理（可选） ======
    # with torch.no_grad():
    #     corr_fixed = getattr(args, "_corr_fixed", None)
    #     a_in, v_in = x[0], x[1]
    #     if corr_fixed is not None:
    #         a_in, v_in = diffusion_guided_denoise(a_in, v_in, corr_fixed, args)
    #         x = (a_in, v_in)
    with torch.no_grad():
        corr_fixed = getattr(args, "_corr_fixed", None)
        a_in, v_in = x[0], x[1]

        # === Anticorruption defiltering ===
        if getattr(args, "audio_corr_name", None):
            a_in = apply_defilter(a_in, args.audio_corr_name)
        if getattr(args, "video_corr_name", None):
            v_in = apply_defilter(v_in, args.video_corr_name)

        # === Optionally keep diffusion-guided ===
        if corr_fixed is not None:
            a_in, v_in = diffusion_guided_denoise(a_in, v_in, corr_fixed, args)

        x = (a_in, v_in)


    # ====== B) 取“原始”单模态 anchor 做一次判定（用于 DUDA 的门控） ======
    a_raw, v_raw = x[0], x[1]
    logits_a_raw, _ = model.module.forward_eval(a=a_raw, v=None,  mode='audio',  args=args, stage='test')
    logits_v_raw, _ = model.module.forward_eval(a=None,  v=v_raw, mode='visual', args=args, stage='test')

    pa_raw = torch.softmax(logits_a_raw.float(), dim=1).clamp_min(eps)
    pv_raw = torch.softmax(logits_v_raw.float(), dim=1).clamp_min(eps)
    C  = pa_raw.size(1); logC = math.log(C + 1e-12)
    Ha = -(pa_raw * pa_raw.log()).sum(dim=1).mean().item() / logC
    Hv = -(pv_raw * pv_raw.log()).sum(dim=1).mean().item() / logC
    corrupted = 'audio' if (Ha - Hv) > float(getattr(args,'tau_ent_hi',0.10)) else \
                'video' if (Hv - Ha) > float(getattr(args,'tau_ent_hi',0.10)) else 'none'

    # ====== C) 对“被判定的那一侧”做 DUDA，再提 anchor ======
    a_anchor_in, v_anchor_in = duda_apply_for_anchor(a_raw, v_raw, which=corrupted, args=args)
    logits_a_anchor, _ = model.module.forward_eval(a=a_anchor_in, v=None,  mode='audio',  args=args, stage='test')
    logits_v_anchor, _ = model.module.forward_eval(a=None,  v=v_anchor_in, mode='visual', args=args, stage='test')

    # ====== D) 懒校准：若尚未校准固定腐败模态 & 微调集合 ======
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
            # 若 none 则保持全冻

            # 与校准一致性：把当前批按 corr 做一次同样的输入级预处理
            a_in2, v_in2 = diffusion_guided_denoise(x[0], x[1], corr, args)
            x = (a_in2, v_in2)

            # 切换后重算 anchors 更稳
            logits_a_anchor, _ = core.forward_eval(a=x[0], v=None,  mode='audio',  args=args, stage='test')
            logits_v_anchor, _ = core.forward_eval(a=None,  v=x[1], mode='visual', args=args, stage='test')

    corr_fixed = getattr(args, "_corr_fixed", "none")

    # ====== E) 若 none：不适配，直接 test-stage 返回 ======
    if corr_fixed == "none":
        with torch.no_grad():
            out_test = core.forward_eval(
                a=x[0], v=x[1], mode=args.testmode, args=args,
                stage='test', labels=labels, ret_av_feat=False, output_embedding=False
            )
        out_test = out_test[0] if isinstance(out_test, (list, tuple)) else out_test
        zero = torch.tensor(0.0, device=out_test.device)
        return (out_test, out_test), (zero.item(), zero.item(), zero.item(), zero.item(), zero.item())

    # ====== F) 正式适配（带图） ======
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

    # ====== G) MLA 权重 ======
    q_a, q_v = compute_modality_weights_mla(
        logits_a_anchor, logits_v_anchor, tau=getattr(args, 'mla_tau', 1.0)
    )
    gap_val = float((q_a - q_v).abs().mean().item())

    # ====== H) 核心损失 ======
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

    # 一致性（PoE）
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

    # ====== I) 总损失 & 反传 ======
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

    # ====== J) 报告前向（test-stage） ======
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
    仍然收集 QKV + 所有Norm的weight/bias（是否更新由 requires_grad 决定）
    """
    params, names = [], []
    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p); names.append(f"{nm}.{np}")
    core = model.module if hasattr(model, "module") else model
    for mod in core.modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            for pn, p in mod.named_parameters(recurse=False):
                if pn in ("weight", "bias"):
                    params.append(p); names.append(f"{type(mod).__name__}.{pn}")
    print(f"[INIT] Collected candidate FT params: {len(params)} tensors.")
    return params, names

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









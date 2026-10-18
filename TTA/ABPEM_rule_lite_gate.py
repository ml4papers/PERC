# ABPEM.py  ———— JS-MLA + JS-corruption-detection + selective LN/QKV unfreezing
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import random
import numpy as np
import torch.nn.functional as F

# ===============================
# DUDA helpers (保留, 可选)
# ===============================
@torch.no_grad()
def _spec_to_4d(S: torch.Tensor, args):
    desc = {"was4d": False, "order": "FT"}
    if S is None:
        return None, desc
    if S.dim() == 4:
        desc["was4d"] = True
        return S, desc
    B, D1, D2 = S.shape
    nbins = int(getattr(args, "num_mel_bins", getattr(args, "mel_bins", 128)))
    if D1 == nbins or (D1 < D2 and D1 <= 256):
        X = S; desc["order"] = "FT"
    elif D2 == nbins or (D2 < D1 and D2 <= 256):
        X = S.transpose(1, 2).contiguous(); desc["order"] = "TF"
    else:
        if D1 <= D2:
            X = S; desc["order"] = "FT"
        else:
            X = S.transpose(1, 2).contiguous(); desc["order"] = "TF"
    return X.unsqueeze(1), desc

@torch.no_grad()
def _spec_from_4d(X4: torch.Tensor, desc):
    if desc.get("was4d", False):
        return X4
    X = X4.squeeze(1)
    if desc.get("order", "FT") == "FT":
        return X
    else:
        return X.transpose(1, 2).contiguous()

@torch.no_grad()
def duda_denoise_audio_anchor(S: torch.Tensor, args):
    if S is None:
        return None
    X4, desc = _spec_to_4d(S, args)
    if X4 is None: return S
    Y = torch.nn.functional.avg_pool2d(X4, kernel_size=3, stride=1, padding=1)
    return _spec_from_4d(Y, desc)

@torch.no_grad()
def duda_denoise_visual_feat(V: torch.Tensor, args):
    if V is None:
        return None
    if V.dim() == 4:
        return torch.nn.functional.avg_pool2d(V, kernel_size=3, stride=1, padding=1)
    else:
        return V

# =========================
# JS-based MLA 权重
# =========================
@torch.no_grad()
def js_div(p, q, eps: float = 1e-6):
    m = 0.5 * (p + q)
    return 0.5 * torch.sum(p * (p.log() - m.log()), dim=1) + \
           0.5 * torch.sum(q * (q.log() - m.log()), dim=1)

@torch.no_grad()
def compute_modality_weights_js(logits_f, logits_a, logits_v, tau: float = 1.0, eps: float = 1e-6):
    p_f = torch.softmax(logits_f, dim=1).clamp_min(eps)
    p_a = torch.softmax(logits_a, dim=1).clamp_min(eps)
    p_v = torch.softmax(logits_v, dim=1).clamp_min(eps)

    d_a = js_div(p_f, p_a)   # [B]
    d_v = js_div(p_f, p_v)   # [B]

    w_a = torch.exp(-d_a / tau)
    w_v = torch.exp(-d_v / tau)
    w = torch.stack([w_a, w_v], dim=1)
    w = w / (w.sum(dim=1, keepdim=True) + eps)
    return w[:,0:1], w[:,1:2]   # q_a, q_v

# =========================
# Corruption Detection (JS-based)
# =========================
@torch.no_grad()
def detect_corrupted_modality(model, loader, args, max_samples: int = 100):
    """
    用 JS divergence 检测哪个模态更 corrupted
    """
    core = model.module if hasattr(model, "module") else model
    device = next(core.parameters()).device
    n_seen = 0
    acc_pf = acc_pa = acc_pv = None

    for a_input, v_input, _ in loader:
        a_input = a_input.to(device, non_blocking=True)
        v_input = v_input.to(device, non_blocking=True)

        out_f = core.forward_eval(a=a_input, v=v_input, mode=args.testmode, args=args, stage='test')
        logits_f = out_f[0] if isinstance(out_f, (list, tuple)) else out_f
        p_f = torch.softmax(logits_f.float(), dim=1).clamp_min(1e-6)

        logits_a, _ = core.forward_eval(a=a_input, v=None, mode='audio', args=args, stage='test')
        logits_v, _ = core.forward_eval(a=None, v=v_input, mode='visual', args=args, stage='test')
        p_a = torch.softmax(logits_a.float(), dim=1).clamp_min(1e-6)
        p_v = torch.softmax(logits_v.float(), dim=1).clamp_min(1e-6)

        b = p_f.size(0)
        if acc_pf is None:
            acc_pf, acc_pa, acc_pv = p_f.mean(0), p_a.mean(0), p_v.mean(0)
        else:
            acc_pf = (acc_pf * n_seen + p_f.sum(0)) / (n_seen + b)
            acc_pa = (acc_pa * n_seen + p_a.sum(0)) / (n_seen + b)
            acc_pv = (acc_pv * n_seen + p_v.sum(0)) / (n_seen + b)
        n_seen += b
        if n_seen >= max_samples:
            break

    if n_seen == 0:
        return "none"

    pf, pa, pv = acc_pf.unsqueeze(0), acc_pa.unsqueeze(0), acc_pv.unsqueeze(0)
    d_a = js_div(pf, pa).mean().item()
    d_v = js_div(pf, pv).mean().item()
    if d_a > d_v:
        return "audio"
    elif d_v > d_a:
        return "video"
    else:
        return "none"

# =========================
# Cache
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

# =========================
# READ wrapper
# =========================
class READ(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
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
                    outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode, args=args)
                loss_tuple = (torch.tensor(0.0),)*5
            return outputs, loss_tuple

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)



# import torch
# import torch.nn.functional as F
#
# @torch.no_grad()
# def adaptive_pre_filter(x: torch.Tensor, blur_thresh=0.05, noise_thresh=0.25):
#     """
#     自适应预滤波:
#       - Noise   → avg-pool denoise
#       - Blur    → unsharp (Laplacian增强)
#       - Digital → median filter
#       - Else    → identity
#     Args:
#       x: Tensor [B,C,H,W]
#     """
#     if x is None or x.dim() != 4:
#         return x
#
#     B, C, H, W = x.shape
#     device, dtype = x.device, x.dtype
#     x_proc = []
#
#     # ---- 定义卷积核 ----
#     lap_kernel = torch.tensor(
#         [[0,-1,0],[-1,4,-1],[0,-1,0]],
#         device=device, dtype=dtype
#     ).view(1,1,3,3).repeat(C,1,1,1)
#
#     for i in range(B):
#         xi = x[i:i+1]  # [1,C,H,W]
#
#         # ---- FFT 分析 ----
#         fft = torch.fft.fft2(xi.float())
#         fft_shift = torch.fft.fftshift(fft)
#         mag = fft_shift.abs()
#
#         # 半径掩码 (low vs high)
#         yy, xx = torch.meshgrid(
#             torch.arange(H, device=device),
#             torch.arange(W, device=device),
#             indexing="ij"
#         )
#         center_y, center_x = H//2, W//2
#         dist = ((yy-center_y)**2 + (xx-center_x)**2).sqrt()
#
#         low_mask  = (dist < min(H,W) * 0.15).float()
#         high_mask = (dist > min(H,W) * 0.65).float()
#
#         low_energy  = (mag * low_mask).mean()
#         high_energy = (mag * high_mask).mean()
#         ratio = (high_energy / (low_energy + 1e-6)).item()
#         ratio = min(ratio, 10.0)  # 防止爆炸
#
#         # ---- filter 选择 ----
#         if ratio > noise_thresh:
#             # 噪声: 平滑
#             yi = F.avg_pool2d(xi, kernel_size=3, stride=1, padding=1)
#         # elif ratio < blur_thresh:
#         #     # 模糊: unsharp
#         #     lap = F.conv2d(xi, lap_kernel, padding=1, groups=C)
#         #     yi = xi + 0.5 * lap
#         # elif blur_thresh <= ratio <= noise_thresh:
#         #     # Digital: median
#         #     unfold = F.unfold(xi, kernel_size=3, padding=1)  # [1, C*9, H*W]
#         #     unfold = unfold.view(1, C, 9, H*W)
#         #     yi = unfold.median(dim=2)[0].view(1, C, H, W)
#         else:
#             yi = xi  # identity
#
#         x_proc.append(yi)
#
#     return torch.cat(x_proc, dim=0)

# @torch.no_grad()
# def adaptive_pre_filter(x: torch.Tensor):
#     """
#     Adaptive pre-filtering:
#       - Blur    → unsharp (Laplacian)
#       - Digital → gaussian + unsharp
#       - Else    → identity
#     """
#     if x is None or x.dim() != 4:
#         return x
#
#     B, C, H, W = x.shape
#     device, dtype = x.device, x.dtype
#     x_proc, ratios = [], []
#
#     lap_kernel = torch.tensor(
#         [[0,-1,0],[-1,4,-1],[0,-1,0]],
#         device=device, dtype=dtype
#     ).view(1,1,3,3).repeat(C,1,1,1)
#
#     for i in range(B):
#         xi = x[i:i+1]
#
#         # ---- FFT ----
#         fft = torch.fft.fft2(xi.float())
#         mag = torch.fft.fftshift(fft).abs()
#
#         yy, xx = torch.meshgrid(
#             torch.arange(H, device=device),
#             torch.arange(W, device=device),
#             indexing="ij"
#         )
#         dist = ((yy-H//2)**2 + (xx-W//2)**2).sqrt()
#
#         low_mask  = (dist < min(H,W)*0.05).float()
#         high_mask = (dist > min(H,W)*0.25).float()
#
#         low_energy  = (mag * low_mask).mean()
#         high_energy = (mag * high_mask).mean()
#         ratio = torch.clamp(high_energy / (low_energy + 1e-6), 0, 10).item()
#         ratios.append(ratio)
#
#         # ---- select filter ----
#         noise_thresh = np.percentile(ratios_hist, 75)
#         if ratio > noise_thresh:
#             yi = F.avg_pool2d(xi, 3, 1, 1)  # 去噪
#
#         # elif ratio < blur_thresh:
#         #     lap = F.conv2d(xi, lap_kernel, padding=1, groups=C)
#         #     yi = xi + 0.5 * lap            # 去模糊
#         # else:
#         #     smooth = F.avg_pool2d(xi, 3, 1, 1)
#         #     lap = F.conv2d(smooth, lap_kernel, padding=1, groups=C)
#         #     yi = smooth + 0.3 * lap        # 去digital
#         else:
#              yi = xi
#         x_proc.append(yi)
#
#     return torch.cat(x_proc, dim=0), ratios



# @torch.no_grad()
# def adaptive_pre_filter(x: torch.Tensor, top_ratio: float = 0.1):
#     """
#     自适应预滤波:
#       - 对能量最高的 top_ratio (默认10%) 频点做平滑滤波
#       - 其他频点保持不变
#     Args:
#       x: [B,C,H,W]
#     """
#     if x is None or x.dim() != 4:
#         return x
#
#     B, C, H, W = x.shape
#     device, dtype = x.device, x.dtype
#     x_proc = []
#
#     for i in range(B):
#         xi = x[i:i+1]  # [1,C,H,W]
#
#         # ---- FFT ----
#         fft = torch.fft.fft2(xi.float())
#         mag = torch.fft.fftshift(fft).abs()   # [1,C,H,W]
#
#         # 单通道平均频谱
#         mag_mean = mag.mean(dim=1, keepdim=True)  # [1,1,H,W]
#         flat = mag_mean.flatten()
#
#         # 取 top k (能量前 10%)
#         k = int(flat.numel() * top_ratio)
#         thresh = torch.topk(flat, k).values.min()
#
#         # 高频 mask
#         high_mask = (mag_mean >= thresh).float()  # [1,1,H,W]
#
#         # 在高频区域做平滑
#         smooth = F.avg_pool2d(xi, kernel_size=3, stride=1, padding=1)
#         yi = xi * (1 - high_mask) + smooth * high_mask  # 混合
#
#         x_proc.append(yi)
#
#     return torch.cat(x_proc, dim=0)

##final version
# @torch.no_grad()
# def adaptive_pre_filter(x: torch.Tensor):
#     """
#     自适应预滤波 (精简稳定版):
#       - 高频能量高 (noise dominated) → median filter
#       - 其他情况 → identity
#     Args:
#       x: Tensor [B,C,H,W]
#     """
#     if x is None or x.dim() != 4:
#         return x
#
#     B, C, H, W = x.shape
#     x_proc,ratios = [],[]
#
#     for i in range(B):
#         xi = x[i:i+1]  # [1,C,H,W]
#
#         # ---- FFT 分析 ----
#         fft = torch.fft.fft2(xi.float())
#         fft_shift = torch.fft.fftshift(fft)
#         mag = fft_shift.abs()
#
#         # 噪声：3x3 median filter
#         unfold = F.unfold(xi, kernel_size=3, padding=1)  # [1, C*9, H*W]
#         unfold = unfold.view(1, C, 9, H*W)
#         median, _ = unfold.median(dim=2)
#         yi = median.view(1, C, H, W)
#
#         x_proc.append(yi)
#
#     return torch.cat(x_proc, dim=0)
##final version - robust to audio
@torch.no_grad()
def adaptive_pre_filter(x: torch.Tensor):
    """
    自适应预滤波 (支持 audio/video):
      - 输入: [B,F,T] (audio spectrogram) 或 [B,C,H,W] (image/video frame)
      - 高频能量高 (noise dominated) → median filter
      - 其他情况 → identity
    """
    if x is None:
        return x

    squeeze_back = False
    if x.dim() == 3:  # audio: [B,F,T] -> [B,1,F,T]
        x = x.unsqueeze(1)
        squeeze_back = True
    elif x.dim() != 4:
        return x

    B, C, H, W = x.shape
    x_proc = []

    for i in range(B):
        xi = x[i:i+1]  # [1,C,H,W]

        # ---- FFT 分析 ----
        fft = torch.fft.fft2(xi.float())
        fft_shift = torch.fft.fftshift(fft)
        mag = fft_shift.abs()

        # ---- 3x3 median filter ----
        unfold = F.unfold(xi, kernel_size=3, padding=1)  # [1, C*9, H*W]
        unfold = unfold.view(1, C, 9, H*W)
        median = unfold.median(dim=2).values             # [1,C,H*W]
        yi = median.view(1, C, H, W)                     # [1,C,H,W]

        x_proc.append(yi)

    out = torch.cat(x_proc, dim=0)

    if squeeze_back:  # 还原回 [B,F,T]
        out = out.squeeze(1)

    return out



# @torch.no_grad()
# def adaptive_pre_filter(x: torch.Tensor, margin=0.1):
#     """
#     Unified corruption-agnostic pre-filter with optional dynamic threshold.
#
#     Args:
#         x: Tensor [B,C,H,W]
#         dynamic: if True, adapt threshold per batch
#         base_thresh: fallback threshold if dynamic=False
#         margin: offset added to median ratio when dynamic=True
#     """
#     if x is None or x.dim() != 4:
#         return x
#
#     B, C, H, W = x.shape
#     ratios, x_proc = [], []
#
#     # Step 1: collect ratios
#     for i in range(B):
#         xi = x[i:i+1]
#
#         fft = torch.fft.fft2(xi.float())
#         mag = torch.fft.fftshift(fft).abs()
#
#         hfreq = mag[:, :, H//4:3*H//4, W//4:3*W//4]
#         lfreq = mag[:, :, :H//8, :W//8]
#
#         high_energy = hfreq.mean()
#         low_energy  = lfreq.mean()
#         ratio = (high_energy / (low_energy + 1e-6)).item()
#         ratios.append(ratio)
#
#     # Step 2: decide threshold
#     noise_thresh = torch.median(torch.tensor(ratios)).item() + margin
#
#     # Step 3: filtering
#     for i, ratio in enumerate(ratios):
#         xi = x[i:i+1]
#         if ratio > noise_thresh:
#             unfold = F.unfold(xi, kernel_size=3, padding=1)
#             unfold = unfold.view(1, C, 9, H*W)
#             median, _ = unfold.median(dim=2)
#             yi = median.view(1, C, H, W)
#         else:
#             yi = xi
#         x_proc.append(yi)
#
#     return torch.cat(x_proc, dim=0)


# @torch.no_grad()
# def blind_deconv_filter(x: torch.Tensor, eps: float = 1e-3, alpha: float = 0.8):
#     """
#     简化版 Blind Deconvolution (FFT域逆滤波近似)
#     Args:
#         x: Tensor [B,C,H,W] (image-like tensor)
#         eps: 数值稳定项，防止除零
#         alpha: 去卷积强度 (0~1)
#     Returns:
#         deblurred tensor, same shape as x
#     """
#     if x is None or x.dim() != 4:
#         return x
#
#     B, C, H, W = x.shape
#     device, dtype = x.device, x.dtype
#     outs = []
#
#     for i in range(B):
#         xi = x[i:i+1]  # [1,C,H,W]
#
#         # ---- FFT ----
#         fft = torch.fft.fft2(xi.float())
#         fft_shift = torch.fft.fftshift(fft)
#
#         # ---- 模拟 blur kernel 的频率响应 (低通) ----
#         yy, xx = torch.meshgrid(
#             torch.arange(H, device=device),
#             torch.arange(W, device=device),
#             indexing="ij"
#         )
#         cy, cx = H // 2, W // 2
#         dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
#         cutoff = min(H, W) * 0.25  # 假设模糊核大小
#         kernel_fft = torch.exp(-(dist**2) / (2 * (cutoff**2)))  # Gaussian LPF
#
#         # ---- Wiener-like 逆滤波 ----
#         kernel_fft = kernel_fft.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
#         kernel_fft = kernel_fft.expand(1, C, H, W)
#         deconv_fft = fft_shift / (kernel_fft + eps)
#
#         # ---- 逆FFT ----
#         deconv = torch.fft.ifft2(torch.fft.ifftshift(deconv_fft)).real
#         # yi = (1 - alpha) * xi + alpha * deconv  # 残差增强
#         yi = deconv
#         outs.append(yi)
#
#     return torch.cat(outs, dim=0).to(dtype)


@torch.no_grad()
def adaptive_pre_filter_audio(S: torch.Tensor, alpha=0.1):
    """
    Lightweight denoise for audio spectrograms (1D smoothing along time).
    Args:
        S: [B,C,F,T] or [B,F,T]
        alpha: smoothing strength
    """
    if S is None:
        return S
    if S.dim() == 3:  # [B,F,T]
        S = S.unsqueeze(1)

    B, C, F, T = S.shape
    # avg_pool along time axis only
    smooth = F.avg_pool2d(S, kernel_size=(1,3), stride=1, padding=(0,1))
    out = S * (1 - alpha) + smooth * alpha

    if out.shape[1] == 1:
        out = out.squeeze(1)
    return out

import torch
import torch.nn.functional as F

@torch.no_grad()
def wiener_on_spectrogram(S: torch.Tensor,
                          assume_power: bool = False,
                          q: float = 0.1,         # 用第 q 分位数估计噪声
                          ema: float = 0.9,       # 掩码时间平滑系数
                          m_floor: float = 0.15,  # 掩码下限，防过抑制
                          eps: float = 1e-6):
    """
    S: [B,F,T] 或 [B,1,F,T] 的幅度谱 (默认) 或功率谱 (assume_power=True)
    返回与 S 同形的去噪谱
    """
    squeeze = False
    if S.dim() == 4 and S.shape[1] == 1:  # [B,1,F,T] -> [B,F,T]
        S = S[:,0]
        squeeze = True
    assert S.dim() == 3, "Expect [B,F,T] or [B,1,F,T]"

    # 1) 功率谱
    P = S if assume_power else (S**2)

    # 2) 每频点噪声估计：时间维 q 分位数（对强非平稳可改滑窗分位数）
    Nhat = torch.quantile(P, q, dim=2, keepdim=True)  # [B,F,1]

    # 3) Wiener 掩码 + 地板
    M = torch.clamp((P - Nhat).relu() / (P + eps), min=m_floor, max=1.0)  # [B,F,T]

    # 4) 时间 EMA 平滑（减小“乐音噪声”）
    M_smooth = []
    for b in range(M.size(0)):
        m = M[b]  # [F,T]
        acc = m[:, :1]
        out = [acc]
        for t in range(1, m.size(1)):
            acc = ema * acc + (1 - ema) * m[:, t:t+1]
            out.append(acc)
        M_smooth.append(torch.cat(out, dim=1))
    M = torch.stack(M_smooth, dim=0)  # [B,F,T]

    # 5) 应用掩码
    if assume_power:
        Sout = M * S
    else:
        Sout = torch.sqrt(torch.clamp(M, min=0.0)) * S

    if squeeze:
        Sout = Sout.unsqueeze(1)  # 还原 [B,1,F,T]
    return Sout

@torch.no_grad()
def wiener_filter_audio(S: torch.Tensor, alpha=0.95):
    """
    Simple Wiener-like filter in spectrogram domain
    Args:
        S: [B,F,T] or [B,1,F,T]
        alpha: smoothing factor
    """
    if S.dim() == 3:
        S = S.unsqueeze(1)

    power = S**2
    smoothed = F.avg_pool2d(power, (1,5), stride=1, padding=(0,2))
    H = smoothed / (smoothed + (1 - alpha))
    out = S * H.sqrt()  # apply gain
    if out.shape[1] == 1:
        out = out.squeeze(1)
    return out






# def adaptive_pre_filter(x: torch.Tensor, base_thresh=0.45, percentile=90):
#     """
#     自适应预滤波 (动态阈值 + soft filtering):
#       - ratio 高 → median filter (强度随 ratio 自适应)
#       - ratio 低 → 保持原样
#     """
#     if x is None or x.dim() != 4:
#         return x
#
#     B, C, H, W = x.shape
#     ratios, x_proc = [], []
#
#     # ---- Step 1: 计算所有 ratio ----
#     for i in range(B):
#         xi = x[i:i+1]
#         fft = torch.fft.fft2(xi.float())
#         mag = torch.fft.fftshift(fft).abs()
#
#         hfreq = mag[:, :, H//4:3*H//4, W//4:3*W//4]
#         lfreq = mag[:, :, :H//8, :W//8]
#
#         ratio = (hfreq.mean() / (lfreq.mean() + 1e-6)).item()
#         ratios.append(ratio)
#
#     # ---- Step 2: 动态阈值 ----
#     dynamic_thresh = np.percentile(ratios, percentile)
#
#     # ---- Step 3: 滤波 ----
#     for i, ratio in enumerate(ratios):
#         xi = x[i:i+1]
#
#         # median filter
#         unfold = F.unfold(xi, kernel_size=3, padding=1)
#         unfold = unfold.view(1, C, 9, H*W)
#         median, _ = unfold.median(dim=2)
#         median = median.view(1, C, H, W)
#
#         # soft filtering 权重
#         alpha = min(1.0, max(0.0, (ratio - dynamic_thresh) / dynamic_thresh))
#         yi = (1 - alpha) * xi + alpha * median
#         x_proc.append(yi)
#
#     return torch.cat(x_proc, dim=0)
    # return torch.cat(x_proc, dim=0), ratios, dynamic_thresh

#
#
# @torch.no_grad()
# def adaptive_pre_filter_audio(S: torch.Tensor, noise_thresh=0.45):
#     """
#     自适应预滤波 for audio spectrograms:
#       - 高频能量高 (noise dominated) → median filter
#       - 否则 → identity
#     Args:
#       S: Tensor [B,C,F,T] (spectrogram)
#     """
#     if S is None or S.dim() != 4:
#         return S
#
#     B, C, F, T = S.shape
#     S_proc = []
#
#     for i in range(B):
#         Si = S[i:i+1]  # [1,C,F,T]
#
#         # ---- FFT 分析 ----
#         fft = torch.fft.fft2(Si.float())
#         fft_shift = torch.fft.fftshift(fft)
#         mag = fft_shift.abs()
#
#         # 高频/低频能量
#         hfreq = mag[:, :, F//4:3*F//4, T//4:3*T//4]
#         lfreq = mag[:, :, :F//8, :T//8]
#
#         high_energy = hfreq.mean()
#         low_energy  = lfreq.mean()
#         ratio = (high_energy / (low_energy + 1e-6)).item()
#
#         # ---- Filter 选择 ----
#         if ratio > noise_thresh:
#             # 噪声: 3x3 median filter
#             unfold = F.unfold(Si, kernel_size=(3,3), padding=1)  # [1, C*9, F*T]
#             unfold = unfold.view(1, C, 9, F*T)
#             median, _ = unfold.median(dim=2)
#             Yi = median.view(1, C, F, T)
#         else:
#             Yi = Si  # identity
#
#         S_proc.append(Yi)
#
#     return torch.cat(S_proc, dim=0)

import torch
import torch.nn.functional as F

@torch.no_grad()
def adaptive_pre_filter_audio(S: torch.Tensor, alpha: float = 0.08):
    """
    Lightweight 1D temporal smoothing for audio spectrograms.
    Args:
        S: [B,F,T] or [B,1,F,T] tensor (spectrogram)
        alpha: smoothing strength (0~0.1 recommended)
    Returns:
        same shape as input
    """
    if S is None:
        return S

    squeeze_back = False
    if S.dim() == 3:   # [B,F,T] → [B,1,F,T]
        S = S.unsqueeze(1)
        squeeze_back = True

    # 沿时间轴做 1D 平滑: kernel=(1,3)
    smooth = F.avg_pool2d(S, kernel_size=(1,3), stride=1, padding=(0,1))
    # 插值混合：保持 α 很小，避免破坏谐波结构
    out = (1 - alpha) * S + alpha * smooth

    if squeeze_back:
        out = out.squeeze(1)   # 还原回 [B,F,T]
    return out

import torch
import torch.nn.functional as F

@torch.no_grad()
def spectral_gating_denoise(S: torch.Tensor, noise_reduction: float = 0.5):
    """
    Spectral gating denoise for audio spectrograms (very lightweight).
    Args:
        S: Tensor [B,F,T] or [B,1,F,T]  (spectrogram, magnitude or log-mel)
        noise_reduction: scaling factor for noise floor (0~1)
    Returns:
        Denoised spectrogram, same shape as input
    """
    if S is None:
        return S
    if S.dim() == 3:   # [B,F,T] -> [B,1,F,T]
        S = S.unsqueeze(1)

    B, C, F, T = S.shape
    out = []

    for i in range(B):
        Si = S[i]  # [C,F,T]

        # ---- Step 1: Estimate noise profile ----
        # 用前10帧估计静音噪声 (也可以 random sample 一段)
        noise_profile = Si[:, :, :min(10, T)].mean(dim=-1, keepdim=True)  # [C,F,1]

        # ---- Step 2: Build mask ----
        mask = (Si > noise_profile * (1.0 + noise_reduction)).float()

        # ---- Step 3: Apply gating ----
        Yi = Si * mask
        out.append(Yi.unsqueeze(0))

    out = torch.cat(out, dim=0)  # [B,C,F,T]

    if out.shape[1] == 1:
        out = out.squeeze(1)  # [B,F,T]

    return out

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def js_div(p, q, eps: float = 1e-6):
    m = 0.5 * (p + q)
    return 0.5 * torch.sum(p * (p.log() - m.log()), dim=1) + \
           0.5 * torch.sum(q * (q.log() - m.log()), dim=1)

@torch.no_grad()
def js_guided_median_filter(x: torch.Tensor,
                            logits_f: torch.Tensor,
                            logits_a: torch.Tensor,
                            logits_v: torch.Tensor,
                            mode: str,
                            k: float = 5.0,
                            tau: float = 0.1):
    """
    JS-guided median filter for both audio & video modality.

    Args:
      x: input tensor
         - audio: [B,F,T] or [B,1,F,T]
         - video: [B,C,H,W]
      logits_f: fused logits [B,num_classes]
      logits_a: audio logits [B,num_classes]
      logits_v: video logits [B,num_classes]
      mode: "audio" or "video"
      k: steepness for sigmoid
      tau: threshold offset for JS → α mapping
    Returns:
      filtered x (same shape)
    """
    if x is None:
        return None

    # ---- Step 1: compute JS divergence ----
    p_f = torch.softmax(logits_f, dim=1).clamp_min(1e-6)
    p_a = torch.softmax(logits_a, dim=1).clamp_min(1e-6)
    p_v = torch.softmax(logits_v, dim=1).clamp_min(1e-6)

    if mode == "audio":
        js_score = js_div(p_f, p_a)  # [B]
    elif mode == "video":
        js_score = js_div(p_f, p_v)  # [B]
    else:
        raise ValueError(f"Unsupported mode={mode}")

    # ---- Step 2: map JS → α ----
    # α in [0,1]; α≈0 means identity, α≈1 means pure median
    alpha = torch.sigmoid(k * (js_score - tau))  # [B]

    # ---- Step 3: apply median filter ----
    squeeze_back = False
    if mode == "audio" and x.dim() == 3:   # [B,F,T]
        x = x.unsqueeze(1)                 # [B,1,F,T]
        squeeze_back = True

    B, C, H, W = x.shape
    outs = []
    for i in range(B):
        xi = x[i:i+1]  # [1,C,H,W]
        patches = F.unfold(xi, kernel_size=3, padding=1)   # [1,C*9,H*W]
        patches = patches.view(1, C, 9, H*W)
        med = patches.median(dim=2).values                 # [1,C,H*W]
        med = med.view(1, C, H, W)

        # weighted mix
        ai = alpha[i].view(1,1,1,1)
        yi = (1 - ai) * xi + ai * med
        outs.append(yi)

    y = torch.cat(outs, dim=0)

    if squeeze_back:
        y = y.squeeze(1)  # [B,F,T]

    return y

@torch.no_grad()
def spectral_subtract_filter(S: torch.Tensor, noise_floor: float = 0.05):
    """
    Simple spectral subtraction baseline
    Args:
        S: [B,F,T] or [B,1,F,T]
        noise_floor: minimum allowed floor (避免负数)
    """
    if S.dim() == 3:
        S = S.unsqueeze(1)

    noise_est = S.mean(dim=-1, keepdim=True)  # 平均噪声
    S_denoised = torch.clamp(S - noise_est, min=noise_floor)
    if S_denoised.shape[1] == 1:
        S_denoised = S_denoised.squeeze(1)
    return S_denoised


@torch.no_grad()
def rule_lite_gate(x: torch.Tensor):
    """
    RAPSD-inspired rule-lite gate:
      - For noise (↑ high-f power) → median filter
      - For blur (↓ high-f power) → unsharp filter
      - Otherwise → identity

    Supports both:
      - Audio spectrogram: [B,F,T]
      - Video/image: [B,C,H,W]
    """
    if x is None:
        return x

    # --- Audio branch ---
    if x.dim() == 3:   # [B,F,T]
        B, Fr, T = x.shape
        outs = []
        for i in range(B):
            xi = x[i:i+1].unsqueeze(1)  # [1,1,F,T]
            fft = torch.fft.fft2(xi.float())
            mag = torch.fft.fftshift(fft).abs()
            hfreq = mag[:, :, Fr//4:3*Fr//4, T//4:3*T//4].mean()
            lfreq = mag[:, :, :Fr//8, :T//8].mean()
            ratio = (hfreq / (lfreq + 1e-6)).item()
            if ratio > 1.2:  # noise → median
                unfold = F.unfold(xi, kernel_size=(3,3), padding=1)
                unfold = unfold.view(1, 1, 9, Fr*T)
                med = unfold.median(dim=2).values.view(1,1,Fr,T)
                yi = med
            else:
                yi = xi
            outs.append(yi)
        out = torch.cat(outs, dim=0).squeeze(1)  # [B,F,T]
        return out

    # --- Video branch ---
    elif x.dim() == 4:  # [B,C,H,W]
        B,C,H,W = x.shape
        outs = []
        for i in range(B):
            xi = x[i:i+1]
            fft = torch.fft.fft2(xi.float())
            mag = torch.fft.fftshift(fft).abs()
            hfreq = mag[:,:,H//4:3*H//4,W//4:3*W//4].mean()
            lfreq = mag[:,:,:H//8,:W//8].mean()
            ratio = (hfreq / (lfreq+1e-6)).item()
            if ratio > 1.2:  # noise
                unfold = F.unfold(xi, kernel_size=3, padding=1)
                unfold = unfold.view(1, C, 9, H*W)
                med = unfold.median(dim=2).values.view(1,C,H,W)
                yi = med
            else:  # blur or neutral
                lap = torch.tensor(
                    [[0,-1,0],[-1,4,-1],[0,-1,0]],
                    device=xi.device, dtype=xi.dtype
                ).view(1,1,3,3).repeat(C,1,1,1)
                lap_out = F.conv2d(xi, lap, padding=1, groups=C)
                yi = xi + 0.3*lap_out
            outs.append(yi)
        return torch.cat(outs, dim=0)

    else:
        raise ValueError(f"Unsupported input shape {x.shape}, expected [B,F,T] or [B,C,H,W].")











# =========================
# forward_and_adapt (JS-MLA)
# =========================
@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    core = model.module if hasattr(model, "module") else model
    eps = 1e-6
    a_raw, v_raw = x[0], x[1]

    # ---------- Step 1: Detect corrupted modality ----------
    corrupted = getattr(args, "_corrupted_modality", "none")
    if corrupted == "audio":
        # a_raw = adaptive_pre_filter(a_raw)
        # a_raw = a_raw
        # a_raw = spectral_gating_denoise(a_raw, noise_reduction=0.3)
        # a_raw = adaptive_pre_filter_audio(a_raw, alpha=-0.1)
        # a_raw = wiener_on_spectrogram(a_raw)
        # a_raw = wiener_filter_audio(a_raw)
        # a_raw = spectral_subtract_filter(a_raw)
        a_raw = rule_lite_gate(a_raw)
    elif corrupted == "video":
        # v_raw, ratios = adaptive_pre_filter(v_raw)
        # v_raw = adaptive_pre_filter(v_raw)
        # v_raw = blind_deconv_filter(v_raw)
        v_raw = rule_lite_gate(v_raw)

    # --- 单模态 anchor logits (不要 detach)
    logits_a_anchor, _ = core.forward_eval(a=a_raw, v=None, mode='audio', args=args, stage='test')
    logits_v_anchor, _ = core.forward_eval(a=None, v=v_raw, mode='visual', args=args, stage='test')

    # --- 融合前向 (student 分支, 可训练)
    with autocast():
        out_pack = core.forward_eval(
            a=a_raw, v=v_raw,
            mode=args.testmode, args=args,
            stage='adapt', labels=labels,
            ret_av_feat=True, output_embedding=True
        )
    outputs, attn_raw, cached_av_feat = out_pack if isinstance(out_pack, (list, tuple)) else (out_pack, None, None)

    # --- JS-MLA 权重
    q_a, q_v = compute_modality_weights_js(outputs, logits_a_anchor, logits_v_anchor,
                                           tau=getattr(args, 'mla_tau', 1.0))

    # --- 损失函数 (四项保留)
    pred = outputs.softmax(dim=-1)
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)
    C = p_sorted.size(1)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)

    # 1) RA loss
    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam_eff = max(1e-6, float(lam0))
    w = torch.exp(-lam_eff * rank)
    w = w / (w.sum() + 1e-12)
    ent_sorted = -p_sorted * torch.log(p_sorted + 1e-6)
    loss_ra = (ent_sorted * w.unsqueeze(0)).sum(dim=-1).mean()

    # 2) Balance loss
    p_sum = pred.mean(dim=0)
    loss_bal = - (p_sum * p_sum.log()).sum()

    # 3) CE loss (pseudo-labels)
    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = F.cross_entropy(outputs, pseudo)

    # 4) Consistency loss
    log_t = q_a * torch.log_softmax(logits_a_anchor, dim=1) + q_v * torch.log_softmax(logits_v_anchor, dim=1)
    t = torch.softmax(log_t, dim=1).clamp_min(eps)
    loss_cons = torch.sum(pred * (pred.add(eps).log() - t.log()), dim=1).mean()

    # 总损失
    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
            + loss_ra * getattr(args, "ent_weight", 0.0)
            - loss_bal * getattr(args, "bal_weight", 0.0)
            + loss_cons * getattr(args, "cons_weight", 0.2))

    # --- 反传优化
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None],
        1.0
    )
    scaler.step(optimizer)
    scaler.update()

    # --- 再 forward 一次 test
    with torch.no_grad():
        out2 = core.forward_eval(
            a=a_raw, v=v_raw, mode=args.testmode, args=args,
            stage='test', labels=labels, av_feat=cached_av_feat
        )
    outputs2 = out2[0] if isinstance(out2, (list, tuple)) else out2

    return (outputs, outputs2), (ce_loss.item(), loss_ra.item(), loss_bal.item(),
                                 loss_cons.item(), loss.item())



# =========================
# collect params & configure
def collect_params(model):
    """
    收集所有 requires_grad=True 的 QKV + Norm 参数
    """
    params, names = [], []   # ✅ 注意这里要两个 list
    core = model.module if hasattr(model, "module") else model

    # --- Fusion QKV ---
    for nm, m in core.named_modules():
        if _name_in_list(nm, FUSION_QKV_NAMES):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias") and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pn}")

    # --- Norm ---
    for nm, mod in core.named_modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            for pn, p in mod.named_parameters(recurse=False):
                if pn in ("weight", "bias") and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pn}")

    total = sum(p.numel() for p in params)
    print(f"[INIT] Collected candidate FT params: {len(params)} tensors (~{total/1e6:.3f} M)")
    return params, names



requires_grad_list = [
    'module.blocks_u.0.attn.qa', 'module.blocks_u.0.attn.ka', 'module.blocks_u.0.attn.va',
    'module.blocks_u.0.attn.qv', 'module.blocks_u.0.attn.kv', 'module.blocks_u.0.attn.vv'
]
def unfreeze_modality_params(model, modality="audio"):
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            m.requires_grad_(True)

    return model




# =========================
# 常量 & 名称匹配工具
# =========================
FUSION_QKV_NAMES = [
    "blocks_u.0.attn.qa", "blocks_u.0.attn.ka", "blocks_u.0.attn.va",
    "blocks_u.0.attn.qv", "blocks_u.0.attn.kv", "blocks_u.0.attn.vv",
    "blocks_u.0.attn.qkv",   # 如果 fusion block 是合并 QKV 结构
]

def _strip_module_prefix(name: str) -> str:
    return name[7:] if name.startswith("module.") else name

def _name_in_list(nm: str, targets: list) -> bool:
    nm2 = _strip_module_prefix(nm)
    return any(nm2.endswith(t) for t in targets)

def configure_model(model):
    """
    初始：冻结所有参数，不做解冻
    在 corruption detection 之后再调用 unfreeze_modality_params 来选择性解冻
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    print("[INIT] all QKV & all norms are frozen; waiting for selective unfreeze...")
    return model










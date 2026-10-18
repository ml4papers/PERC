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

# # =========================
# # READ wrapper
# # =========================
# class READ(nn.Module):
#     def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
#         super().__init__()
#         self.model = model
#         self.optimizer = optimizer
#         self.steps = steps
#         self.episodic = episodic
#         self.args = args
#         self.scaler = GradScaler()
#         self.device = device
#
#         self.args._ema_m = getattr(args, "ema_m", 0.999)
#         self.args._ema_model = deepcopy(self.model).to(device).eval()
#         for p in self.args._ema_model.parameters():
#             p.requires_grad_(False)
#
#     def forward(self, x, adapt_flag, labels=None):
#         for _ in range(self.steps):
#             if adapt_flag:
#                 outputs, loss_tuple = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler, labels=labels)
#             else:
#                 args = self.args
#                 with torch.no_grad():
#                     outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode, args=args)
#                 loss_tuple = (torch.tensor(0.0),)*5
#             return outputs, loss_tuple
# =========================
# READ wrapper
# =========================
class READ(nn.Module):
    def __init__(self, model, optimizer, device, args,
                 steps=1, episodic=False,
                 audio_filter=None, video_filter=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

        # 可选滤波器（learnable module）
        self.audio_filter = audio_filter
        self.video_filter = video_filter

        # === EMA copy for stable eval ===
        self.args._ema_m = getattr(args, "ema_m", 0.999)
        self.args._ema_model = deepcopy(self.model).to(device).eval()
        for p in self.args._ema_model.parameters():
            p.requires_grad_(False)

    def forward(self, x, adapt_flag, labels=None):
        a_raw, v_raw = x

        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss_tuple = forward_and_adapt(
                    (a_raw, v_raw),
                    self.model,
                    self.optimizer,
                    self.args,
                    self.scaler,
                    labels=labels,
                    audio_filter=self.audio_filter,   # ✅ 传进去
                    video_filter=self.video_filter    # ✅ 传进去
                )
            else:
                args = self.args
                with torch.no_grad():
                    outputs, _ = self.model.module.forward_eval(
                        a=a_raw, v=v_raw, mode=args.testmode, args=args
                    )
                loss_tuple = (torch.tensor(0.0, device=self.device),) * 5
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
@torch.no_grad()
def adaptive_pre_filter(x: torch.Tensor):
    """
    自适应预滤波 (精简稳定版):
      - 高频能量高 (noise dominated) → median filter
      - 其他情况 → identity
    Args:
      x: Tensor [B,C,H,W]
    """
    if x is None or x.dim() != 4:
        return x

    B, C, H, W = x.shape
    x_proc,ratios = [],[]

    for i in range(B):
        xi = x[i:i+1]  # [1,C,H,W]

        # ---- FFT 分析 ----
        fft = torch.fft.fft2(xi.float())
        fft_shift = torch.fft.fftshift(fft)
        mag = fft_shift.abs()

        # 噪声：3x3 median filter
        unfold = F.unfold(xi, kernel_size=3, padding=1)  # [1, C*9, H*W]
        unfold = unfold.view(1, C, 9, H*W)
        median, _ = unfold.median(dim=2)
        yi = median.view(1, C, H, W)

        x_proc.append(yi)

    return torch.cat(x_proc, dim=0)


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

@torch.no_grad()
def adaptive_pre_filter_audio(S: torch.Tensor, alpha=0.1):
    """
    Lightweight denoise for audio spectrograms (1D smoothing along time).
    Args:
        S: [B,C,F,T] or [B,F,T]
        alpha: smoothing strength
    """
    print('audio',S.shape)
    if S is None:
        return S
    if S.dim() == 3:  # [B,F,T]
        S = S.unsqueeze(1)

    # avg_pool along time axis only
    smooth = F.avg_pool2d(S, kernel_size=(1,3), stride=1, padding=(0,1))
    out = S * (1 - alpha) + smooth * alpha

    if out.shape[1] == 1:
        out = out.squeeze(1)
    return out

class LearnableMedianFilter(nn.Module):
    def __init__(self, channels=1, kernel_size=3, per_channel=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.per_channel = per_channel
        if per_channel:
            self.alpha = nn.Parameter(torch.zeros(channels))
        else:
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 输入 [B,F,T] -> 升维 [B,1,F,T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, C, H, W = x.shape

        unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        unfold = unfold.view(B, C, self.kernel_size ** 2, H * W)
        median, _ = unfold.median(dim=2)
        median = median.view(B, C, H, W)

        alpha = torch.sigmoid(self.alpha)
        if self.per_channel:
            alpha = alpha.view(1, -1, 1, 1)
        else:
            alpha = alpha.view(1, 1, 1, 1)

        y = (1 - alpha) * x + alpha * median

        # === 保持和输入一致 ===
        if y.shape[1] == 1:
            y = y.squeeze(1)  # 回到 [B,F,T]

        return y


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





# # =========================
# # forward_and_adapt (JS-MLA)
# # =========================
# @torch.enable_grad()
# def forward_and_adapt(x, model, optimizer, args, scaler, labels=None,
#                       audio_filter=None, video_filter=None):
#     core = model.module if hasattr(model, "module") else model
#     a_raw, v_raw = x[0], x[1]
#
#     # === Apply learnable filters ===
#     if audio_filter is not None and a_raw is not None:
#         a_raw = audio_filter(a_raw)
#     if video_filter is not None and v_raw is not None:
#         v_raw = video_filter(v_raw)
#
#
#
#     # forward
#
#     # --- 单模态 anchor logits (不要 detach)
#     logits_a_anchor, _ = core.forward_eval(a=a_raw, v=None, mode='audio', args=args, stage='test')
#     logits_v_anchor, _ = core.forward_eval(a=None, v=v_raw, mode='visual', args=args, stage='test')
#
#     # --- 融合前向 (student 分支, 可训练)
#     with autocast():
#         out_pack = core.forward_eval(
#             a=a_raw, v=v_raw,
#             mode=args.testmode, args=args,
#             stage='adapt', labels=labels,
#             ret_av_feat=True, output_embedding=True
#         )
#     outputs, attn_raw, cached_av_feat = out_pack if isinstance(out_pack, (list, tuple)) else (out_pack, None, None)
#
#     # --- JS-MLA 权重
#     q_a, q_v = compute_modality_weights_js(outputs, logits_a_anchor, logits_v_anchor,
#                                            tau=getattr(args, 'mla_tau', 1.0))
#
#     # --- 损失函数 (四项保留)
#     pred = outputs.softmax(dim=-1)
#     p_sorted, _ = torch.sort(pred, dim=-1, descending=True)
#     C = p_sorted.size(1)
#     rank = torch.arange(C, device=pred.device, dtype=pred.dtype)
#
#     # 1) RA loss
#     lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
#     lam_eff = max(1e-6, float(lam0))
#     w = torch.exp(-lam_eff * rank)
#     w = w / (w.sum() + 1e-12)
#     ent_sorted = -p_sorted * torch.log(p_sorted + 1e-6)
#     loss_ra = (ent_sorted * w.unsqueeze(0)).sum(dim=-1).mean()
#
#     # 2) Balance loss
#     p_sum = pred.mean(dim=0)
#     loss_bal = - (p_sum * p_sum.log()).sum()
#
#     # 3) CE loss (pseudo-labels)
#     with torch.no_grad():
#         pseudo = outputs.argmax(dim=1)
#     ce_loss = F.cross_entropy(outputs, pseudo)
#
#     # 4) Consistency loss
#     log_t = q_a * torch.log_softmax(logits_a_anchor, dim=1) + q_v * torch.log_softmax(logits_v_anchor, dim=1)
#     t = torch.softmax(log_t, dim=1).clamp_min(eps)
#     loss_cons = torch.sum(pred * (pred.add(eps).log() - t.log()), dim=1).mean()
#
#     # 总损失
#     loss = (ce_loss * getattr(args, "ce_weight", 1.0)
#             + loss_ra * getattr(args, "ent_weight", 0.0)
#             - loss_bal * getattr(args, "bal_weight", 0.0)
#             + loss_cons * getattr(args, "cons_weight", 0.2))
#
#     # --- 反传优化
#     optimizer.zero_grad(set_to_none=True)
#     scaler.scale(loss).backward()
#     scaler.unscale_(optimizer)
#     torch.nn.utils.clip_grad_norm_(
#         [p for p in model.parameters() if p.requires_grad and p.grad is not None],
#         1.0
#     )
#     scaler.step(optimizer)
#     scaler.update()
#
#     # --- 再 forward 一次 test
#     with torch.no_grad():
#         out2 = core.forward_eval(
#             a=a_raw, v=v_raw, mode=args.testmode, args=args,
#             stage='test', labels=labels, av_feat=cached_av_feat
#         )
#     outputs2 = out2[0] if isinstance(out2, (list, tuple)) else out2
#
#     return (outputs, outputs2), (ce_loss.item(), loss_ra.item(), loss_bal.item(),
#                                  loss_cons.item(), loss.item())

# =========================
# forward_and_adapt (JS-MLA + filters)
# =========================
@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None,
                      audio_filter=None, video_filter=None):
    core = model.module if hasattr(model, "module") else model
    a_raw, v_raw = x[0], x[1]   # a_raw: [B,F,T]  v_raw: [B,C,H,W]

    # === Apply learnable filters ===
    if audio_filter is not None and a_raw is not None:
        # audio spectrogram [B,F,T] -> [B,1,F,T]
        # audio spectrogram [B,F,T] -> [B,1,F,T]
        print("before", a_raw.shape)
        a_raw = audio_filter(a_raw)  # [B,1,F,T]
        print("after", a_raw.shape)

    if video_filter is not None and v_raw is not None:
        v_raw = video_filter(v_raw)  # already [B,C,H,W]

    # --- 单模态 anchor logits ---
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

    # --- 损失函数 ---
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

    # 3) CE loss
    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = F.cross_entropy(outputs, pseudo)

    # 4) Consistency loss
    eps = 1e-6
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
# def collect_params(model):
#     """
#     收集所有 requires_grad=True 的 QKV + Norm 参数
#     """
#     params, names = [], []   # ✅ 注意这里要两个 list
#     core = model.module if hasattr(model, "module") else model
#
#     # --- Fusion QKV ---
#     for nm, m in core.named_modules():
#         if _name_in_list(nm, FUSION_QKV_NAMES):
#             for pn, p in m.named_parameters(recurse=False):
#                 if pn in ("weight", "bias") and p.requires_grad:
#                     params.append(p)
#                     names.append(f"{nm}.{pn}")
#
#     # --- Norm ---
#     for nm, mod in core.named_modules():
#         if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
#             for pn, p in mod.named_parameters(recurse=False):
#                 if pn in ("weight", "bias") and p.requires_grad:
#                     params.append(p)
#                     names.append(f"{nm}.{pn}")
#
#     total = sum(p.numel() for p in params)
#     print(f"[INIT] Collected candidate FT params: {len(params)} tensors (~{total/1e6:.3f} M)")
#     return params, names
def collect_params(model, audio_filter=None, video_filter=None):
    """
    收集所有可训练的 QKV + Norm (LN/GN/BN) 参数 + 可学习滤波器参数
    """
    params, names = [], []
    core = model.module if hasattr(model, "module") else model

    # --- QKV 层 ---
    for nm, m in core.named_modules():
        if _name_in_list(nm, FUSION_QKV_NAMES):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias") and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pn}")

    # --- Norm 层 (LN/GN/BN) ---
    for mod in core.modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            for pn, p in mod.named_parameters(recurse=False):
                if pn in ("weight", "bias") and p.requires_grad:
                    params.append(p)
                    names.append(f"{type(mod).__name__}.{pn}")

    # --- Filters (audio / video) ---
    if audio_filter is not None:
        for pn, p in audio_filter.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(f"audio_filter.{pn}")

    if video_filter is not None:
        for pn, p in video_filter.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(f"video_filter.{pn}")

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

# def configure_model(model):
#     """
#     初始：冻结所有参数，不做解冻
#     在 corruption detection 之后再调用 unfreeze_modality_params 来选择性解冻
#     """
#     model.train()
#     for p in model.parameters():
#         p.requires_grad = False
#     print("[INIT] all QKV & all norms are frozen; waiting for selective unfreeze...")
#     return model
def configure_model(model, extra_modules=None):
    model.train()
    for p in model.parameters():
        p.requires_grad = False

    if extra_modules is not None:
        for em in extra_modules:
            for p in em.parameters():
                p.requires_grad = True

    print("[INIT] backbone frozen; filter params kept trainable")
    return model












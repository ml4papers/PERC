# ABPEM.py — JS-MLA + JS-corruption-detection + selective LN/QKV unfreezing
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
import random
import numpy as np


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
# =========================
# JS-based MLA 权重
# =========================
EPS = 1e-6
@torch.no_grad()
# def js_div(p, q, eps: float = 1e-6):
#     m = 0.5 * (p + q)
#     return 0.5 * torch.sum(p * (p.log() - m.log()), dim=1) + \
#            0.5 * torch.sum(q * (q.log() - m.log()), dim=1)

# def kl_pq(p, q, eps=EPS):
# def js_div(p, q, eps=EPS):
#     return torch.sum(p * (torch.log(p+eps) - torch.log(q+eps)), dim=1)

# def kl_qp(p, q, eps=EPS):  # reverse-KL: KL(q||p)
def js_div(p, q, eps=EPS):
    return torch.sum(q * (torch.log(q+eps) - torch.log(p+eps)), dim=1)
#
# def cosine_distance(p, q, eps=EPS):
#     # 1 - cosine similarity
#     num = torch.sum(p * q, dim=1)
#     den = torch.linalg.norm(p, dim=1) * torch.linalg.norm(q, dim=1) + eps
#     return 1.0 - num / den

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
# Pre-filters
# =========================
@torch.no_grad()
def adaptive_pre_filter_audio(S: torch.Tensor, alpha: float = 0.08):
    """轻量 1D 平滑 for audio spectrograms"""
    if S is None:
        return S
    squeeze_back = False
    if S.dim() == 3:   # [B,F,T] → [B,1,F,T]
        S = S.unsqueeze(1); squeeze_back = True
    smooth = F.avg_pool2d(S, kernel_size=(1,3), stride=1, padding=(0,1))
    out = (1 - alpha) * S + alpha * smooth
    if squeeze_back:
        out = out.squeeze(1)
    return out

@torch.no_grad()
def spectral_gating_denoise(S: torch.Tensor, noise_reduction: float = 0.5):
    """简单谱门控去噪"""
    if S is None: return S
    if S.dim() == 3: S = S.unsqueeze(1)
    B, C, F, T = S.shape
    outs = []
    for i in range(B):
        Si = S[i]  # [C,F,T]
        noise_profile = Si[:, :, :min(10, T)].mean(dim=-1, keepdim=True)
        mask = (Si > noise_profile * (1.0 + noise_reduction)).float()
        outs.append((Si * mask).unsqueeze(0))
    out = torch.cat(outs, dim=0)
    if out.shape[1] == 1: out = out.squeeze(1)
    return out

@torch.no_grad()
def rule_lite_gate(x: torch.Tensor):
    """
    RAPSD-inspired rule-lite gate:
      - Audio [B,F,T]: noise → median
      - Video [B,C,H,W]: noise → median; blur → unsharp
    """
    if x is None: return x

    # Audio branch
    if x.dim() == 3:   # [B,F,T]
        B, Fr, T = x.shape; outs = []
        for i in range(B):
            xi = x[i:i+1].unsqueeze(1)  # [1,1,F,T]
            fft = torch.fft.fft2(xi.float()); mag = torch.fft.fftshift(fft).abs()
            hfreq = mag[:, :, Fr//4:3*Fr//4, T//4:3*T//4].mean()
            lfreq = mag[:, :, :Fr//8, :T//8].mean()
            ratio = (hfreq / (lfreq + 1e-6)).item()
            print('ratio', ratio)
            if ratio > 1.0:  # noise → median
                unfold = F.unfold(xi, kernel_size=(3,3), padding=1).view(1, 1, 9, Fr*T)
                med = unfold.median(dim=2).values.view(1,1,Fr,T); yi = med
            else: yi = xi
            outs.append(yi)
        return torch.cat(outs, dim=0).squeeze(1)

    # Video branch
    elif x.dim() == 4:  # [B,C,H,W]
        B,C,H,W = x.shape; outs = []
        lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]],
                           device=x.device, dtype=x.dtype).view(1,1,3,3).repeat(C,1,1,1)
        for i in range(B):
            xi = x[i:i+1]; fft = torch.fft.fft2(xi.float()); mag = torch.fft.fftshift(fft).abs()
            hfreq = mag[:,:,H//4:3*H//4,W//4:3*W//4].mean()
            lfreq = mag[:,:,:H//8,:W//8].mean()
            ratio = (hfreq / (lfreq+1e-6)).item()
            # print('ratio', ratio)
            if ratio > 1.0:  # noise → median
                unfold = F.unfold(xi, kernel_size=3, padding=1).view(1,C,9,H*W)
                med = unfold.median(dim=2).values.view(1,C,H,W); yi = med
            else:  # blur → unsharp
                lap_out = F.conv2d(xi, lap, padding=1, groups=C); yi = xi + 0.3*lap_out
            outs.append(yi)
        return torch.cat(outs, dim=0)

    else:
        raise ValueError(f"Unsupported shape {x.shape}")

@torch.no_grad()
def calibrate_fft_thresholds(clean_batch_list, modality="video"):
    """
    自动估计 clean 数据下的 FFT 高频/低频能量比阈值。
    输入:
      clean_batch_list: list of [B,C,H,W] 张量
    输出:
      dict: {"mean_ratio": float, "thr_noise": float, "thr_blur": float}
    """
    ratios = []
    for xb in clean_batch_list:
        thr_cfg = {"thr_noise": 1.3, "thr_blur": 0.8}
        dec, ratio, _ = fft_ratio_detect(xb, thr_cfg=thr_cfg, modality=modality)

        ratios.append(ratio)
    ratios = torch.cat(ratios)
    mean_r = ratios.mean().item()
    std_r = ratios.std().item()
    return {
        "mean_ratio": mean_r,
        "thr_noise": mean_r + 2.0 * std_r,
        "thr_blur":  mean_r - 2.0 * std_r
    }

import torch
import torch.nn.functional as F

# =========================================
# FFT-based corruption detection utilities
# =========================================

@torch.no_grad()
def fft_ratio_detect(x: torch.Tensor, thr_cfg=None, modality="video"):
    """
    检测 noise / blur / fog (低对比度)
    返回: (decisions, R, A)
    """
    if thr_cfg is None:
        thr_cfg = {"noise": 1.3, "blur": 0.8, "fog_amp": 0.05}

    # 调整输入形状
    if modality == "audio" and x.dim() == 3:
        x = x.unsqueeze(1)
    elif modality == "video" and x.dim() == 3:
        x = x.unsqueeze(1)

    B, C, H, W = x.shape
    fft = torch.fft.fft2(x.float())
    mag = torch.fft.fftshift(fft).abs().mean(1)  # [B,H,W]

    # 高频 / 低频掩码
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2).to(mag.device)
    r_thr = min(H, W) * 0.25
    high_mask = (r >= r_thr).float()
    low_mask  = (r < r_thr).float()

    high = (mag * high_mask).mean(dim=(-2, -1))
    low  = (mag * low_mask).mean(dim=(-2, -1))
    total_amp = mag.mean(dim=(-2, -1))
    ratio = high / (low + 1e-6)

    decisions = []
    for r_val, a_val in zip(ratio, total_amp):
        r_val = float(r_val)
        a_val = float(a_val)
        if r_val > thr_cfg["noise"]:
            decisions.append("noise")
        elif r_val < thr_cfg["blur"]:
            decisions.append("blur")
        elif a_val < thr_cfg["fog_amp"]:
            decisions.append("fog")
        else:
            decisions.append("clean")

    return decisions, ratio, total_amp


@torch.no_grad()
def calibrate_fft_thresholds(clean_batches, modality="video"):
    """
    动态标定 FFT 阈值（无 clean baseline）
    从若干 batch 的分布统计中位数作为中心
    """
    all_R, all_A = [], []
    for xb in clean_batches:
        _, R, A = fft_ratio_detect(xb, modality=modality)
        all_R.append(R.cpu())
        all_A.append(A.cpu())
    all_R = torch.cat(all_R)
    all_A = torch.cat(all_A)

    med_R = all_R.median().item()
    med_A = all_A.median().item()
    return {
        "noise": med_R * 1.25,   # 高频偏强判为噪声
        "blur":  med_R * 0.85,   # 高频偏弱判为模糊
        "fog_amp": med_A * 0.6   # 振幅偏低判为雾或低对比度
    }


@torch.no_grad()
def fft_filter_adaptive(x: torch.Tensor, decisions, modality="video"):
    """
    根据 corruption 类型自适应修复：
      - noise → median
      - blur  → unsharp
      - fog   → contrast stretch
    """
    outs = []
    for i, dec in enumerate(decisions):
        xi = x[i:i+1]
        if dec == "noise":
            B, C, H, W = xi.shape
            unfold = F.unfold(xi, kernel_size=5, padding=2).view(B, C, 25, H * W)
            med = unfold.median(dim=2).values.view(B, C, H, W)
            yi = med

        # elif dec == "blur":
        #     B, C, H, W = xi.shape
        #     lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        #                        device=xi.device, dtype=xi.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        #     lap_out = F.conv2d(xi, lap, padding=1, groups=C)
        #     yi = xi + 0.3 * lap_out

        elif dec == "fog":
            # 线性对比度增强
            minv = xi.amin(dim=(-2, -1), keepdim=True)
            maxv = xi.amax(dim=(-2, -1), keepdim=True)
            yi = (xi - minv) / (maxv - minv + 1e-6)
            yi = torch.clamp(yi * 1.3, 0, 1)

        else:
            yi = xi  # clean
        outs.append(yi)

    return torch.cat(outs, dim=0)

@torch.no_grad()
def _get_radial_bins(H, W, nbins=64, device=None):
    """
    生成径向频率分布的 bin map（供 RAPSD 使用）
    返回:
      rbin_map: [H,W] 每个像素对应的 bin 索引
      centers: [nbins] bin 中心半径 (归一化)
      counts:  [nbins] 每个 bin 中像素数量
    """
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    cy, cx = H / 2, W / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r / r.max()  # 归一化半径范围 [0,1]

    # 分桶
    bins = torch.linspace(0, 1, nbins + 1, device=device)
    rbin_map = torch.bucketize(r, bins) - 1
    rbin_map = torch.clamp(rbin_map, 0, nbins - 1)

    counts = torch.bincount(rbin_map.view(-1), minlength=nbins).float()
    centers = 0.5 * (bins[:-1] + bins[1:])

    return rbin_map, centers, counts

# 拟合对数功率谱斜率

def _fit_slope_loglog(rad_psd, centers, low=0.1, high=0.9):
    """
    对 log-log 功率谱拟合斜率 α（线性回归）
    rad_psd: [B, nbins]
    centers: [nbins]
    返回: α [B]
    """
    B = rad_psd.size(0)
    nbins = centers.numel()
    idx = (centers >= low) & (centers <= high)
    x = torch.log(centers[idx] + 1e-6)
    X = torch.stack([x, torch.ones_like(x)], dim=1)  # [M,2]

    logP = torch.log(rad_psd[:, idx] + 1e-6)
    pinv = torch.linalg.pinv(X)
    coeff = (logP @ pinv.T)  # [B,2]
    alpha = coeff[:, 0]  # slope
    return alpha

import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def compute_rapsd_curve(x: torch.Tensor, nbins: int = 64):
    """Compute Radially Averaged Power Spectral Density (RAPSD)."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    B, C, H, W = x.shape
    device = x.device
    fft = torch.fft.fft2(x.float())
    psd = (fft.abs() ** 2).mean(dim=1)  # [B,H,W]
    rbin_map, centers, counts = _get_radial_bins(H, W, nbins, device)
    idx = rbin_map.flatten()
    vals = psd.view(B, -1)
    rad_sum = torch.zeros(B, nbins, device=device)
    rad_sum.scatter_add_(1, idx.unsqueeze(0).expand(B, -1), vals)
    rad_psd = rad_sum / (counts.unsqueeze(0) + 1e-6)
    return centers.cpu(), rad_psd.mean(dim=0).cpu()


import os
import random
from PIL import Image
from torchvision import transforms

@torch.no_grad()
def load_clean_samples_from_dir(dir_path, num_samples=4, resize=224, device="cuda"):
    """
    Recursively load random clean images from subdirectories of dir_path (e.g., frames0, frames1, ...).
    Args:
        dir_path: root directory (can contain multiple frame folders)
        num_samples: number of images to load
        resize: resize to HxW
    Returns:
        Tensor [B,3,H,W] normalized to [0,1]
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Clean path not found: {dir_path}")

    # Recursively find all image files
    image_files = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, f))

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found recursively in {dir_path}")

    # Randomly sample a few
    selected = random.sample(image_files, min(num_samples, len(image_files)))

    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])

    imgs = [transform(Image.open(p).convert("RGB")) for p in selected]
    imgs = torch.stack(imgs).to(device)
    print(f"[LOAD] Loaded {len(imgs)} clean samples from {dir_path}")
    return imgs



@torch.no_grad()
def plot_rapsd_three_panel(clean_img, corruption_groups, filtered_sets,
                           save_path="rapsd_three_panel.png"):
    """
    Plot three subplots:
      (a) All 15 corruption + clean
      (b) Noise types (3) + clean + filtered
      (c) Contrast + clean + filtered
    Args:
        clean_img: [B,C,H,W] clean tensor
        corruption_groups: dict {
            "all": {"Gaussian": x1, "shot": x2, ..., "contrast": xN},
            "noise": {"Gaussian": xg, "shot": xs, "impulse": xi},
            "contrast": {"contrast": xc}
        }
        filtered_sets: dict {
            "noise_filtered": x_nf,
            "contrast_filtered": x_cf
        }
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)
    plt.subplots_adjust(wspace=0.25)
    colors = plt.cm.tab20.colors

    # === Subplot (a): all 15 corrupted + clean ===
    ax = axes[0]
    c_src, r_src = compute_rapsd_curve(clean_img)
    ax.plot(c_src, r_src, "k", linewidth=2, label="source")
    for i, (name, img) in enumerate(corruption_groups["all"].items()):
        c, r = compute_rapsd_curve(img)
        ax.plot(c, r, color=colors[i % len(colors)], linewidth=1.2, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequency"); ax.set_ylabel("RAPSD")
    ax.set_title("(a) All Corruptions")
    ax.legend(fontsize=6, ncol=2, loc="lower left")

    # === Subplot (b): noise + clean + filtered ===
    ax = axes[1]
    c_src, r_src = compute_rapsd_curve(clean_img)
    ax.plot(c_src, r_src, "k", linewidth=2, label="source")
    for i, (name, img) in enumerate(corruption_groups["noise"].items()):
        c, r = compute_rapsd_curve(img)
        ax.plot(c, r, color=colors[i], linewidth=1.2, label=name)
    if "noise_filtered" in filtered_sets:
        c_f, r_f = compute_rapsd_curve(filtered_sets["noise_filtered"])
        ax.plot(c_f, r_f, "b--", linewidth=1.8, label="filtered")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequency")
    ax.set_title("(b) Noise + Filtered")
    ax.legend(fontsize=7, loc="lower left")

    # === Subplot (c): contrast + clean + filtered ===
    ax = axes[2]
    c_src, r_src = compute_rapsd_curve(clean_img)
    ax.plot(c_src, r_src, "k", linewidth=2, label="source")
    c_con, r_con = compute_rapsd_curve(corruption_groups["contrast"]["contrast"])
    ax.plot(c_con, r_con, "r", linewidth=1.2, label="contrast")
    if "contrast_filtered" in filtered_sets:
        c_cf, r_cf = compute_rapsd_curve(filtered_sets["contrast_filtered"])
        ax.plot(c_cf, r_cf, "b--", linewidth=1.8, label="filtered")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequency")
    ax.set_title("(c) Contrast + Filtered")
    ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] RAPSD 3-panel figure saved to {save_path}")
    plt.show()


def torch_interp(x, xp, fp):
    """
    PyTorch版线性插值 (替代 torch.interp)
    x  : 待插值向量
    xp : 已知自变量 (升序)
    fp : 已知因变量
    """
    x = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, x) - 1
    idx = idx.clamp(0, len(xp) - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    y0, y1 = fp[idx], fp[idx + 1]
    slope = (y1 - y0) / (x1 - x0 + 1e-6)
    return y0 + slope * (x - x0)

@torch.no_grad()
def histogram_equalization(x: torch.Tensor, n_bins: int = 256):
    """
    x: [B, C, H, W] tensor, values in [0,1]
    Return: histogram equalized tensor
    """
    B, C, H, W = x.shape
    outs = []
    for i in range(B):
        xi = x[i:i+1]
        flat = xi.view(-1)
        hist = torch.histc(flat, bins=n_bins, min=0.0, max=1.0)
        cdf = hist.cumsum(0)
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-6)
        xp = torch.linspace(0, 1, n_bins, device=x.device)
        fp = cdf
        x_eq = torch_interp(flat, xp, fp)
        outs.append(x_eq.view_as(xi))
    return torch.cat(outs, dim=0)

@torch.no_grad()
def adaptive_histogram_equalization(x: torch.Tensor, grid_size: int = 8, n_bins: int = 256, clip_limit: float = 0.01):
    """
    Simplified CLAHE implementation in PyTorch.
    x: [B, C, H, W] tensor in [0,1]
    """
    B, C, H, W = x.shape
    out = torch.zeros_like(x)
    tile_h, tile_w = H // grid_size, W // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            patch = x[:, :, y0:y1, x0:x1]
            flat = patch.flatten(2)
            hist = torch.histc(flat, bins=n_bins, min=0.0, max=1.0)
            hist = torch.clamp(hist, max=clip_limit * flat.numel())
            cdf = hist.cumsum(0)
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-6)
            xp = torch.linspace(0, 1, n_bins, device=x.device)
            fp = cdf
            x_eq = torch_interp(flat, xp, fp)
            out[:, :, y0:y1, x0:x1] = x_eq.view_as(patch)
    return out


# 用法：



@torch.no_grad()
def rapsd_slope_detector(x: torch.Tensor, modality="video",
                         nbins: int = 64,
                         mid_band=(0.15, 0.4),
                         high_band=(0.6, 0.95),
                         tau: float = 0.2):
    """
    改进版 RAPSD 双区间斜率差检测:
      Δα = α_high - α_mid
      Δα > +τ  → noise
      Δα < -τ  → blur
      否则 clean
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)

    B, C, H, W = x.shape
    device = x.device

    # FFT & 平均功率谱
    fft = torch.fft.fft2(x.float())
    Pm = (fft.abs() ** 2).mean(dim=1)  # [B,H,W]

    # 径向平均功率谱
    rbin_map, centers, counts = _get_radial_bins(H, W, nbins, device)
    idx = rbin_map.flatten()
    vals = Pm.view(B, -1)
    rad_sum = torch.zeros(B, nbins, device=device)
    rad_sum.scatter_add_(1, idx.unsqueeze(0).expand(B, -1), vals)
    rad_psd = rad_sum / (counts.unsqueeze(0) + 1e-6)



    alpha_mid = _fit_slope_loglog(rad_psd, centers)
    alpha_high = _fit_slope_loglog(rad_psd, centers)
    delta_alpha = alpha_high - alpha_mid

    # 判别
    dec = []
    for da in delta_alpha:
        da_val = da.item()
        if da_val > tau:
            dec.append("noise")
        elif da_val < -tau:
            dec.append("blur")
        else:
            dec.append("clean")

    return dec, delta_alpha








# # =========================
# # forward_and_adapt
# # =========================
# @torch.enable_grad()
# def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
#     core = model.module if hasattr(model, "module") else model
#     eps = 1e-6; a_raw, v_raw = x[0], x[1]
#
#     # Step 1: Anchors
#     logits_a_anchor,_ = core.forward_eval(a=a_raw,v=None,mode='audio',args=args,stage='test')
#     logits_v_anchor,_ = core.forward_eval(a=None,v=v_raw,mode='visual',args=args,stage='test')
#     logits_f = core.forward_eval(a=a_raw,v=v_raw,mode=args.testmode,args=args,stage='test')[0]
#
#     # Step 2: Corruption detection (EMA)
#     if not hasattr(args,"_ema_d_a"): args._ema_d_a,args._ema_d_v=0.0,0.0
#     d_a = js_div(logits_f.softmax(-1), logits_a_anchor.softmax(-1)).mean().item()
#     d_v = js_div(logits_f.softmax(-1), logits_v_anchor.softmax(-1)).mean().item()
#     args._ema_d_a = 0.9*args._ema_d_a + 0.1*d_a
#     args._ema_d_v = 0.9*args._ema_d_v + 0.1*d_v
#     corrupted = "audio" if args._ema_d_a>args._ema_d_v else "video" if args._ema_d_v>args._ema_d_a else "none"
#
#     # --- Step 3: 如果检测到的 corruption 变了，就 unfreeze + 重新 collect params + 重建 optimizer
#     if not hasattr(args, "_cur_corrupted") or args._cur_corrupted != corrupted:
#         args._cur_corrupted = corrupted
#         print(f"[ADAPT] Switch detected → unfreezing {corrupted} params")
#         unfreeze_modality_params(model, corrupted)
#         params, _ = collect_params(model)
#         optimizer.param_groups.clear()
#         optimizer.add_param_group({"params": params, "lr": args.lr, "weight_decay": args.wd})
#
#
#     # Step 3: Pre-filter
#     if corrupted == "audio":
#         a_raw = rule_lite_gate(a_raw)
#         a_raw = a_raw.detach().clone().requires_grad_()
#     elif corrupted == "video":
#         v_raw = rule_lite_gate(v_raw)
#         v_raw = v_raw.detach().clone().requires_grad_()
#
#     # Step 4: Forward adapt
#     with autocast():
#         out_pack = core.forward_eval(a=a_raw,v=v_raw,mode=args.testmode,args=args,
#                                      stage='adapt',labels=labels,ret_av_feat=True,output_embedding=True)
#     outputs, attn_raw, cached_av_feat = out_pack if isinstance(out_pack,(list,tuple)) else (out_pack,None,None)
#
#     # Step 5: JS-MLA weights
#     q_a,q_v = compute_modality_weights_js(outputs,logits_a_anchor,logits_v_anchor,
#                                           tau=getattr(args,'mla_tau',1.0))
#
#     # Step 6: Losses
#     pred = outputs.softmax(dim=-1); p_sorted,_ = torch.sort(pred,dim=-1,descending=True)
#     C = p_sorted.size(1); rank = torch.arange(C,device=pred.device,dtype=pred.dtype)
#     lam0 = getattr(args,"re_lambda0",getattr(args,"re_lambda",0.0)); lam_eff = max(1e-6,float(lam0))
#     w = torch.exp(-lam_eff*rank); w = w/(w.sum()+1e-12)
#     ent_sorted = -p_sorted*torch.log(p_sorted+1e-6); loss_ra = (ent_sorted*w.unsqueeze(0)).sum(dim=-1).mean()
#     p_sum = pred.mean(dim=0); loss_bal = -(p_sum*p_sum.log()).sum()
#     with torch.no_grad(): pseudo = outputs.argmax(dim=1)
#     ce_loss = F.cross_entropy(outputs,pseudo)
#     log_t = q_a*torch.log_softmax(logits_a_anchor,dim=1)+q_v*torch.log_softmax(logits_v_anchor,dim=1)
#     t = torch.softmax(log_t,dim=1).clamp_min(eps)
#     loss_cons = torch.sum(pred*(pred.add(eps).log()-t.log()),dim=1).mean()
#     loss = (ce_loss*getattr(args,"ce_weight",1.0)
#            +loss_ra*getattr(args,"ent_weight",0.0)
#            -loss_bal*getattr(args,"bal_weight",0.0)
#            +loss_cons*getattr(args,"cons_weight",0.2))
#
#     # Step 7: Backward
#     optimizer.zero_grad(set_to_none=True)
#     scaler.scale(loss).backward(); scaler.unscale_(optimizer)
#     torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None],1.0)
#     scaler.step(optimizer); scaler.update()
#
#     # Step 8: Final test forward
#     with torch.no_grad():
#         out2 = core.forward_eval(a=a_raw,v=v_raw,mode=args.testmode,args=args,
#                                  stage='test',labels=labels,av_feat=cached_av_feat)
#     outputs2 = out2[0] if isinstance(out2,(list,tuple)) else out2
#
#     return (outputs,outputs2),(ce_loss.item(),loss_ra.item(),loss_bal.item(),loss_cons.item(),loss.item())
@torch.no_grad()
def detect_noise_or_contrast(x: torch.Tensor, modality="video",
                                thr_noise=1.3, thr_local=0.08, thr_dr=1.1,
                                thr_std=0.06, thr_ratio=0.9):
    """
    更稳健的 noise / contrast 检测:
      - noise: 高频能量占比高
      - contrast: 局部对比度 & 动态范围低
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    B, C, H, W = x.shape
    x_gray = x.mean(dim=1, keepdim=True)

    # === FFT ratio ===
    fft = torch.fft.fft2(x_gray)
    mag = torch.fft.fftshift(fft).abs().mean(1)
    Hc, Wc = mag.shape[-2:]
    cy, cx = Hc // 2, Wc // 2
    yy, xx = torch.meshgrid(torch.arange(Hc), torch.arange(Wc), indexing="ij")
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2).to(mag.device)
    r_thr = min(Hc, Wc) * 0.25
    high = (mag * (r >= r_thr)).mean(dim=(-2, -1))
    low  = (mag * (r <  r_thr)).mean(dim=(-2, -1))
    ratio = high / (low + 1e-6)

    # === 空域统计 ===
    std_vals = x_gray.std(dim=(2, 3))
    # 局部对比度（滑动窗口 std）
    unfold = F.unfold(x_gray, kernel_size=11, padding=5)
    local_std = unfold.std(dim=1).mean(dim=1)
    # 全局动态范围
    drange = (x_gray.amax(dim=(2, 3)) - x_gray.amin(dim=(2, 3))) / (x_gray.mean(dim=(2, 3)) + 1e-6)

    decisions = []
    for i in range(B):
        if ratio[i] > thr_noise:
            decisions.append("noise")
        elif (local_std[i] < thr_local and drange[i] < thr_dr) or (ratio[i] < thr_ratio and std_vals[i] < thr_std):
            decisions.append("contrast")
        else:
            decisions.append("clean")

    return decisions, ratio, std_vals, local_std, drange


@torch.no_grad()
def filter_noise_or_contrast(x: torch.Tensor, decisions, modality="video"):
    """
    针对 noise / contrast 的对应修复
    """
    outs = []
    for i, dec in enumerate(decisions):
        xi = x[i:i+1]
        if dec == "noise":
            # 中值滤波
            B, C, H, W = xi.shape
            unfold = F.unfold(xi, kernel_size=5, padding=2).view(B, C, 25, H * W)
            med = unfold.median(dim=2).values.view(B, C, H, W)
            outs.append(med)
        elif dec == "contrast":
            # 自适应对比度增强
            x_min = xi.amin(dim=(2, 3), keepdim=True)
            x_max = xi.amax(dim=(2, 3), keepdim=True)
            x_norm = (xi - x_min) / (x_max - x_min + 1e-6)
            x_enh = torch.clamp((x_norm - 0.5) * 1.8 + 0.5, 0, 1)
            outs.append(x_enh)
            # x_enh = adaptive_histogram_equalization(xi)
            # x_enh = histogram_equalization(xi)
            # outs.append(x_enh)
        else:
            outs.append(xi)
    return torch.cat(outs, dim=0)

def spectral_consistency_loss(x, target_slope=-2.0, nbins=64, band=(0.15, 0.85)):
    """
    ⚙️ 频谱一致性约束（仅用于视觉分支）：
      - 拟合功率谱 log-log 斜率 α
      - 约束 α ≈ target_slope (通常 -2)
      - 作为附加正则项 L_spec = |α - α*| 或平方项

    参数:
        x: [B,C,H,W] 图像 / 视频帧张量
        target_slope: 理论自然图像斜率 (~ -2)
        nbins: 径向采样数量
        band: 拟合的频率范围比例
    返回:
        loss_spec (float Tensor)
        alpha (Tensor[B]) 拟合到的斜率
    """
    if x is None:
        return torch.tensor(0.0, device="cuda"), torch.tensor([0.0], device="cuda")

    B, C, H, W = x.shape
    device = x.device

    # === Step 1. 计算平均功率谱 ===
    fft = torch.fft.fft2(x.float())
    psd = (fft.abs()**2).mean(dim=1)  # [B,H,W]

    # === Step 2. 径向平均 ===
    rbin_map, centers, counts = _get_radial_bins(H, W, nbins, device)
    idx = rbin_map.flatten()
    vals = psd.view(B, -1)
    rad_sum = torch.zeros(B, nbins, device=device)
    rad_sum.scatter_add_(1, idx.unsqueeze(0).expand(B, -1), vals)
    rad_psd = rad_sum / (counts.unsqueeze(0) + 1e-6)

    # === Step 3. log-log 线性拟合斜率 ===
    alpha = _fit_slope_loglog(rad_psd, centers, *band)  # [B]

    # === Step 4. 损失（平方差 or L1）===
    loss_spec = ((alpha - target_slope)**2).mean()

    return loss_spec, alpha

@torch.no_grad()
def detect_audio_noise(S: torch.Tensor, hi_frac=0.25, c=2.0):
    """
    快速检测音频谱图中的噪声：
      - 高频/低频功率比 > 阈值 → noise
    S: [B, 1, F, T] 或 [B, F, T]
    """
    if S.dim() == 3:
        S = S.unsqueeze(1)
    B, C, F, T = S.shape
    fft = torch.fft.fft(S.float(), dim=2)      # 沿频率方向分析
    mag = fft.abs().mean(1)                    # [B,F,T]
    f_thr = int(F * (1 - hi_frac))
    hi = mag[:, f_thr:, :].mean(dim=(1,2))
    lo = mag[:, :f_thr, :].mean(dim=(1,2))
    R = hi / (lo + 1e-6)
    med = R.median()
    mad = (R - med).abs().median()
    thr = med + c * mad
    decisions = (R > thr).to(torch.int32)
    strength = torch.clamp((R - thr) / (thr + 1e-6), 0, 1)
    return decisions, strength, R, thr

@torch.no_grad()
def median_filter_audio(S: torch.Tensor, kf=3, kt=5):
    """Mel-spectrogram median filter: [B,1,F,T]"""
    if S.dim() == 3:
        S = S.unsqueeze(1)
    B, C, Fh, Tw = S.shape
    patches = F.unfold(S, kernel_size=(kf, kt), padding=(kf//2, kt//2))  # [B, C*kf*kt, Fh*Tw]
    patches = patches.view(B, C, kf*kt, Fh*Tw)
    med = patches.median(dim=2).values.view(B, C, Fh, Tw)
    return med



@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    core = model.module if hasattr(model, "module") else model
    eps = 1e-6
    a_raw, v_raw = x[0], x[1]

    # === Step 1: Anchors ===
    logits_a_anchor, _ = core.forward_eval(a=a_raw, v=None, mode='audio', args=args, stage='test')
    logits_v_anchor, _ = core.forward_eval(a=None, v=v_raw, mode='visual', args=args, stage='test')
    logits_f = core.forward_eval(a=a_raw, v=v_raw, mode=args.testmode, args=args, stage='test')[0]


    # === Step 2: Corruption detection (JS-div for modality switch) ===

    # # Instance-wise corruption detection
    # d_a = js_div(logits_f.softmax(-1), logits_a_anchor.softmax(-1), eps=1e-6)
    # d_v = js_div(logits_f.softmax(-1), logits_v_anchor.softmax(-1), eps=1e-6)
    # corrupted = ["audio" if da > dv else "video" for da, dv in zip(d_a, d_v)]
    #
    # Batch-wise corruption detection
    # d_a = js_div(logits_f.softmax(-1), logits_a_anchor.softmax(-1)).mean().item()
    # d_v = js_div(logits_f.softmax(-1), logits_v_anchor.softmax(-1)).mean().item()
    # corrupted = "audio" if d_a > d_v else "video"

    # EMA-smoothed batch-wise detection    if not hasattr(args, "_ema_d_a"):
    args._ema_d_a, args._ema_d_v = 0.0, 0.0
    d_a = js_div(logits_f.softmax(-1), logits_a_anchor.softmax(-1)).mean().item()
    d_v = js_div(logits_f.softmax(-1), logits_v_anchor.softmax(-1)).mean().item()
    args._ema_d_a = 0.9 * args._ema_d_a + 0.1 * d_a
    args._ema_d_v = 0.9 * args._ema_d_v + 0.1 * d_v
    corrupted = "audio" if args._ema_d_a > args._ema_d_v else "video" if args._ema_d_v > args._ema_d_a else "none"


    # === Step 3: 动态 unfreeze + optimizer 更新 ===
    if not hasattr(args, "_cur_corrupted") or args._cur_corrupted != corrupted:
        args._cur_corrupted = corrupted
        print(f"[ADAPT] Switch detected → unfreezing {corrupted} params")
    unfreeze_modality_params(model)
    params, names = collect_params(model)
    if len(params) > 0:
        optimizer.param_groups[0]['params'] = params
        print(f"[ADAPT] optimizer updated with {len(params)} tensors (~{sum(p.numel() for p in params) / 1e6:.3f} M)")
    else:
        print("[WARN] still no trainable params after unfreeze")

    # === Step 4: FFT-based corruption detector + filter ===
    # === Step 4: FFT 检测 + 自适应滤波 ===
    # Step 4: Noise / Contrast detection
    if corrupted == "video":
        dec_v, R_v, std_v, L_v, DR_v = detect_noise_or_contrast(v_raw, modality="video")
        print(f"[DETECT] video → {dec_v[:8]}... | R={R_v.mean():.3f} L={L_v.mean():.3f} DR={DR_v.mean():.3f}")
        v_raw = filter_noise_or_contrast(v_raw, dec_v, modality="video")
        #visualize the rapsd curves
        plot_rapsd_three_panel(
            clean_img=load_clean_samples_from_dir(
                "/home/jonaslei/DuBaTTA/data_path/Kinetics50/image_mulframe_val256_k=50",
                num_samples=4,
                resize=v_raw.shape[-1],
                device=v_raw.device
            ),
            corruption_dict={
                "Gaussian": v_raw, "Shot": v_raw, "Impulse": v_raw,
                "Defocus": v_raw, "Glass": v_raw, "Motion": v_raw, "Zoom": v_raw,
                "Snow": v_raw, "Frost": v_raw, "Fog": v_raw, "Brightness": v_raw,
                "Contrast": v_raw, "Elastic": v_raw, "Pixelate": v_raw, "JPEG": v_raw
            },
            filtered_sets={"noise_filtered": v_raw, "contrast_filtered": v_raw},
            save_path="results/rapsd_three_panel_visual.png"
        )

    # if corrupted == "audio":
    #     dec_a, str_a, R_a, thr_a = detect_audio_noise(a_raw)
    #     if dec_a.sum() > 0:
    #         print(f"[AUDIO] noise detected {dec_a.sum().item()}/{len(dec_a)} "
    #               f"(R={R_a.mean():.3f} thr={thr_a:.3f})")
    #         a_raw = median_filter_audio(a_raw, kf=3, kt=7)

    # === Step 5: Forward adapt ===
    with autocast():
        out_pack = core.forward_eval(a=a_raw.squeeze(), v=v_raw, mode=args.testmode, args=args,
                                     stage='adapt', labels=labels, ret_av_feat=True, output_embedding=True)
    outputs, attn_raw, cached_av_feat = out_pack if isinstance(out_pack, (list, tuple)) else (out_pack, None, None)

    # === Step 6: Losses ===
    pred = outputs.softmax(dim=-1)
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)
    C = p_sorted.size(1)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)
    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam_eff = max(1e-6, float(lam0))
    w = torch.exp(-lam_eff * rank)
    w = w / (w.sum() + 1e-12)
    ent_sorted = -p_sorted * torch.log(p_sorted + 1e-6)
    loss_ra = (ent_sorted * w.unsqueeze(0)).sum(dim=-1).mean()

    p_sum = pred.mean(dim=0)
    loss_bal = -(p_sum * p_sum.log()).sum()

    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = F.cross_entropy(outputs, pseudo)

    log_t = torch.log_softmax(logits_a_anchor, dim=1) * 0.5 + torch.log_softmax(logits_v_anchor, dim=1) * 0.5
    t = torch.softmax(log_t, dim=1).clamp_min(eps)
    loss_cons = torch.sum(pred * (pred.add(eps).log() - t.log()), dim=1).mean()

    # # === Step 6.1: Dynamic normalization (no EMA, per-batch) ===
    # # 初始化一次
    # if not hasattr(args, "norm_stats"):
    #     args.norm_stats = {"ce": 1.0, "rem": 1.0, "bal": 1.0, "cons": 1.0}
    #
    # # 每步更新滑动均值（momentum 可 0.9~0.99）
    # momentum = 0.9
    # for k, v in zip(["ce", "rem", "bal", "cons"],
    #                 [ce_loss.item(), loss_ra.item(), loss_bal.item(), loss_cons.item()]):
    #     args.norm_stats[k] = momentum * args.norm_stats[k] + (1 - momentum) * v
    #
    # eps_n = 1e-6
    # n_ce   = ce_loss   / (args.norm_stats["ce"]   + eps_n)
    # n_rem  = loss_ra   / (args.norm_stats["rem"]  + eps_n)
    # n_bal  = loss_bal  / (args.norm_stats["bal"]  + eps_n)
    # n_cons = loss_cons / (args.norm_stats["cons"] + eps_n)
    #
    # alpha = getattr(args, "ent_weight", 0.4)
    # beta  = getattr(args, "bal_weight", 0.6)
    # gamma = getattr(args, "cons_weight", 0.2)
    #
    # # final total normalized loss
    # loss = n_ce + alpha * n_rem - beta * n_bal + gamma * n_cons

    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
           + loss_ra * getattr(args, "ent_weight", 0.2)
           - loss_bal * getattr(args, "bal_weight", 0.6)
           + loss_cons * getattr(args, "cons_weight", 0.4))

    # === Step 6.5: Spectral Consistency Loss (only for video) ===
    # if corrupted == "video":
    #     loss_spec, alpha_v = spectral_consistency_loss(v_raw, target_slope=-2.0)
    #     loss = loss + getattr(args, "spec_weight", 0.1) * loss_spec
    #     print(f"[SPEC] α={alpha_v.mean():.2f}, L_spec={loss_spec.item():.4f}")
    # else:
    #     loss_spec = torch.tensor(0.0, device=outputs.device)

    # === Step 7: Backward ===
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0)
    if len(optimizer.param_groups) > 0:
        scaler.step(optimizer)
        scaler.update()
    else:
        print("[WARN] optimizer has no param_groups, skip step")

    # === Step 8: Final test forward ===
    with torch.no_grad():
        out2 = core.forward_eval(a=a_raw, v=v_raw, mode=args.testmode, args=args,
                                 stage='test', labels=labels, av_feat=cached_av_feat)
    outputs2 = out2[0] if isinstance(out2, (list, tuple)) else out2

    return (outputs, outputs2), (ce_loss.item(), loss_ra.item(), loss_bal.item(), loss_cons.item(), loss.item())



# =========================
# Param config
# =========================
FUSION_QKV_NAMES = ["blocks_u.0.attn.qa","blocks_u.0.attn.ka","blocks_u.0.attn.va",
                    "blocks_u.0.attn.qv","blocks_u.0.attn.kv","blocks_u.0.attn.vv",
                    "blocks_u.0.attn.qkv"]

def _strip_module_prefix(name: str) -> str:
    return name[7:] if name.startswith("module.") else name

def _name_in_list(nm: str, targets: list) -> bool:
    return any(_strip_module_prefix(nm).endswith(t) for t in targets)

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

def configure_model(model):
    model.train()
    for p in model.parameters(): p.requires_grad=False
    print("[INIT] all QKV & all norms are frozen; waiting for selective unfreeze...")
    return model












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

import torch
import torch.nn.functional as F

# ========== FFT 工具 ==========

@torch.no_grad()
def robust_stats(x: torch.Tensor):
    med = x.median()
    mad = (x - med).abs().median() + 1e-6
    return med, mad

@torch.no_grad()
def calibrate_fft_thresholds(clean_batch_list, modality="video",
                             r_thr_frac=0.25, kN=2.5, kB=2.5, kF=2.0):
    """
    用干净图像批量标定 noise / blur / fog 的检测阈值
    """
    Rs, As = [], []
    for x in clean_batch_list:
        if modality == "video" and x.dim() == 3:
            x = x.unsqueeze(1)
        elif modality == "audio" and x.dim() == 3:
            x = x.unsqueeze(1)

        fft = torch.fft.fft2(x.float())
        mag = torch.fft.fftshift(fft).abs().mean(1)
        H, W = mag.shape[-2:]
        cy, cx = H // 2, W // 2
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        r = torch.sqrt((yy - cy)**2 + (xx - cx)**2).to(mag.device)
        r_thr = min(H, W) * r_thr_frac

        low = (mag * (r < r_thr)).mean(dim=(-2, -1))
        mag_norm = mag / (low.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        high = (mag_norm * (r >= r_thr)).mean(dim=(-2, -1))
        R = high
        A = mag_norm.mean(dim=(-2, -1))
        Rs.append(R.cpu())
        As.append(A.cpu())

    R_all = torch.cat(Rs)
    A_all = torch.cat(As)
    mR, sR = robust_stats(R_all)
    mA, sA = robust_stats(A_all)

    thr = {
        "thr_noise": (mR + kN * sR).item(),
        "thr_blur":  (mR - kB * sR).item(),
        "thr_fog":   (mA - kF * sA).item(),
        "mR": mR.item(), "sR": sR.item(),
        "mA": mA.item(), "sA": sA.item()
    }
    print(f"[CALIB] R_clean={mR:.3f}±{sR:.3f}, A_clean={mA:.3f}±{sA:.3f}")
    print(f"[CALIB] thr_noise={thr['thr_noise']:.3f}, thr_blur={thr['thr_blur']:.3f}, thr_fog={thr['thr_fog']:.3f}")
    return thr


@torch.no_grad()
def fft_ratio_detect(x: torch.Tensor, thr, modality="video", r_thr_frac=0.25):
    if modality == "audio" and x.dim() == 3:
        x = x.unsqueeze(1)
    elif modality == "video" and x.dim() == 3:
        x = x.unsqueeze(1)

    fft = torch.fft.fft2(x.float())
    mag = torch.fft.fftshift(fft).abs().mean(1)
    H, W = mag.shape[-2:]
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2).to(mag.device)
    r_thr = min(H, W) * r_thr_frac

    low = (mag * (r < r_thr)).mean(dim=(-2, -1))
    mag_norm = mag / (low.unsqueeze(-1).unsqueeze(-1) + 1e-6)
    high = (mag_norm * (r >= r_thr)).mean(dim=(-2, -1))
    R = high
    A = mag_norm.mean(dim=(-2, -1))

    decisions = []
    for ri, ai in zip(R, A):
        r_val, a_val = float(ri), float(ai)
        if r_val > thr["thr_noise"]:
            decisions.append("noise")
        elif r_val < thr["thr_blur"]:
            decisions.append("blur")
        elif a_val < thr["thr_fog"]:
            decisions.append("fog")
        else:
            decisions.append("clean")
    return decisions, R, A


@torch.no_grad()
def fft_filter_adaptive(x: torch.Tensor, decisions, modality="video"):
    outs = []
    for i, dec in enumerate(decisions):
        xi = x[i:i+1]
        if dec == "noise":
            # Median filter 去噪
            B, C, H, W = xi.shape
            unfold = F.unfold(xi, kernel_size=5, padding=2).view(B, C, 25, H*W)
            med = unfold.median(dim=2).values.view(B, C, H, W)
            yi = med
        elif dec == "fog":
            # 对比度拉伸
            minv = xi.amin(dim=(-1, -2), keepdim=True)
            maxv = xi.amax(dim=(-1, -2), keepdim=True)
            yi = (xi - minv) / (maxv - minv + 1e-6)
        elif dec == "blur":
            # Unsharp mask 锐化
            B, C, H, W = xi.shape
            lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                               device=xi.device, dtype=xi.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
            lap_out = F.conv2d(xi, lap, padding=1, groups=C)
            yi = xi + 0.3 * lap_out
        else:
            yi = xi
        outs.append(yi)
    return torch.cat(outs, dim=0)






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
    if not hasattr(args, "_ema_d_a"):
        args._ema_d_a, args._ema_d_v = 0.0, 0.0
    d_a = js_div(logits_f.softmax(-1), logits_a_anchor.softmax(-1)).mean().item()
    d_v = js_div(logits_f.softmax(-1), logits_v_anchor.softmax(-1)).mean().item()
    args._ema_d_a = 0.9 * args._ema_d_a + 0.1 * d_a
    args._ema_d_v = 0.9 * args._ema_d_v + 0.1 * d_v
    corrupted = "audio" if args._ema_d_a > args._ema_d_v else "video" if args._ema_d_v > args._ema_d_a else "none"

    # === Step 2.5: 初始化 FFT 阈值（一次性） ===
    if corrupted == "video":
        if not hasattr(args, "_fft_thr"):
            # ⚠️ 用几张干净的样本做一次阈值标定
            clean_batch_list = [v_raw[:2].detach().clone()]  # 例如取当前 batch 前 2 张
            args._fft_thr = calibrate_fft_thresholds(clean_batch_list, modality="video")
            print("[INIT] Calibrated FFT thresholds for corruption detection.")

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
    if corrupted == "video":
        dec_v, R_v, A_v = fft_ratio_detect(v_raw, args._fft_thr, modality="video")
        print(f"[FFT] video detection → {dec_v} | R={R_v.mean():.3f} A={A_v.mean():.3f}")
        v_raw = fft_filter_adaptive(v_raw, dec_v, modality="video")

    # === Step 5: Forward adapt ===
    with autocast():
        out_pack = core.forward_eval(a=a_raw, v=v_raw, mode=args.testmode, args=args,
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

    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
           + loss_ra * getattr(args, "ent_weight", 0.0)
           - loss_bal * getattr(args, "bal_weight", 0.0)
           + loss_cons * getattr(args, "cons_weight", 0.2))

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











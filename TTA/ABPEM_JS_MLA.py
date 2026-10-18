# ABPEM.py  ———— JS-MLA + DUDA + 全LN解冻
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
# DUDA helpers (保留: 你可开关)
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
    # 简单谱平滑 (示例, 可替换更复杂版本)
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

@torch.no_grad()
def duda_apply_for_anchor(a_raw, v_raw, which: str, args):
    if which == 'audio':
        return duda_denoise_audio_anchor(a_raw, args), v_raw
    elif which == 'video':
        return a_raw, duda_denoise_visual_feat(v_raw, args)
    else:
        return a_raw, v_raw

# =========================
# JS-based MLA 权重
# =========================
@torch.no_grad()
def compute_modality_weights_js(logits_f, logits_a, logits_v, tau: float = 1.0, eps: float = 1e-6):
    p_f = torch.softmax(logits_f, dim=1).clamp_min(eps)
    p_a = torch.softmax(logits_a, dim=1).clamp_min(eps)
    p_v = torch.softmax(logits_v, dim=1).clamp_min(eps)

    def js_div(p, q):
        m = 0.5 * (p + q)
        return 0.5 * torch.sum(p * (p.log() - m.log()), dim=1) + \
               0.5 * torch.sum(q * (q.log() - m.log()), dim=1)

    d_a = js_div(p_f, p_a)   # [B]
    d_v = js_div(p_f, p_v)   # [B]

    w_a = torch.exp(-d_a / tau)
    w_v = torch.exp(-d_v / tau)
    w = torch.stack([w_a, w_v], dim=1)
    w = w / (w.sum(dim=1, keepdim=True) + eps)
    return w[:,0:1], w[:,1:2]   # q_a, q_v

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

# =========================
# forward_and_adapt (JS-MLA)
# =========================
@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    core = model.module if hasattr(model, "module") else model
    eps = 1e-6
    a_raw, v_raw = x[0], x[1]

    # 单模态 logits
    logits_a_raw, _ = core.forward_eval(a=a_raw, v=None,  mode='audio',  args=args, stage='test')
    logits_v_raw, _ = core.forward_eval(a=None,  v=v_raw, mode='visual', args=args, stage='test')

    # DUDA anchors
    a_anchor_in, v_anchor_in = duda_apply_for_anchor(a_raw, v_raw, which='none', args=args)
    logits_a_anchor, _ = core.forward_eval(a=a_anchor_in, v=None,  mode='audio',  args=args, stage='test')
    logits_v_anchor, _ = core.forward_eval(a=None,  v=v_anchor_in, mode='visual', args=args, stage='test')

    # 融合前向 (adapt)
    with autocast():
        out_pack = core.forward_eval(
            a=a_raw, v=v_raw,
            mode=args.testmode, args=args,
            stage='adapt', labels=labels,
            ret_av_feat=True, output_embedding=True
        )
    outputs, attn_raw, cached_av_feat = out_pack if isinstance(out_pack, (list,tuple)) else (out_pack,None,None)

    # JS-MLA 权重
    q_a, q_v = compute_modality_weights_js(outputs, logits_a_anchor, logits_v_anchor, tau=getattr(args, 'mla_tau', 1.0))

    # 损失
    pred = outputs.softmax(dim=-1)
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)
    C = p_sorted.size(1)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)

    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam_eff = max(1e-6, float(lam0))
    w = torch.exp(-lam_eff * rank); w = w / (w.sum() + 1e-12)

    ent_sorted = -p_sorted * torch.log(p_sorted + 1e-6)
    loss_ra = (ent_sorted * w.unsqueeze(0)).sum(dim=-1).mean()

    p_sum = pred.mean(dim=0)
    loss_bal = - (p_sum * p_sum.log()).sum()

    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = torch.nn.functional.cross_entropy(outputs, pseudo)

    log_t = q_a * torch.log_softmax(logits_a_anchor, dim=1) + q_v * torch.log_softmax(logits_v_anchor, dim=1)
    t = torch.softmax(log_t, dim=1).clamp_min(eps)
    loss_cons = torch.sum(pred * (pred.add(eps).log() - t.log()), dim=1).mean()

    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
            + loss_ra * getattr(args, "ent_weight", 0.0)
            - loss_bal * getattr(args, "bal_weight", 0.0)
            + loss_cons * getattr(args, "cons_weight", 0.2))

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0)
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        out2 = core.forward_eval(a=a_raw, v=v_raw, mode=args.testmode, args=args, stage='test', labels=labels, av_feat=cached_av_feat)
    outputs2 = out2[0] if isinstance(out2,(list,tuple)) else out2

    return (outputs, outputs2), (ce_loss.item(), loss_ra.item(), loss_bal.item(), loss_cons.item(), loss.item())

# =========================
# collect params & configure
# =========================
def collect_params(model):
    params, names = [], []
    core = model.module if hasattr(model, "module") else model
    for mod in core.modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            for pn, p in mod.named_parameters(recurse=False):
                if pn in ("weight", "bias"):
                    params.append(p); names.append(f"{type(mod).__name__}.{pn}")
    print(f"[INIT] Collected candidate FT params: {len(params)} tensors.")
    return params, names

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

def configure_model(model):
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    print("[INIT] all QKV & all norms are frozen; wait for dynamic unfreeze.")

    # 🔑 新增：全量解冻 QKV
    for nm, m in model.named_modules():
        if _name_in_list(nm, AUDIO_QKV_NAMES + VISUAL_QKV_NAMES):
            for p in m.parameters():
                p.requires_grad = True
            print(f"[FT-init] unfreeze QKV: {nm}")

    # LN/BN 也默认解冻
    for mod in model.modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            for pn, p in mod.named_parameters(recurse=False):
                if pn in ("weight", "bias"):
                    p.requires_grad = True

    return model











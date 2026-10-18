import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np

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

    # ---------- (0) 单模态 anchors（test-stage，无梯度） ----------
    with torch.no_grad():
        logits_a_anchor, _ = core.forward_eval(a=x[0], v=None,  mode='audio',  args=args, stage='test')
        logits_v_anchor, _ = core.forward_eval(a=None,  v=x[1], mode='visual', args=args, stage='test')

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




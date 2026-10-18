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

# --------------------- 判定 + 选择性解冻 LN ---------------------

def _softmax_stable(logits, eps=1e-6):
    return torch.softmax(logits, dim=1).clamp_min(eps)

def _kl_pq(p, q, eps=1e-6):
    # KL(p || q) = sum p (log p - log q)
    return torch.sum(p * (p.add(eps).log() - q.add(eps).log()), dim=1)  # [B]

def decide_corrupted_modality(logits_f, logits_a, logits_v, args):
    """
    返回: 'audio' | 'visual' | None
    规则：比较 KL(p_f||p_a) 与 KL(p_f||p_v)，并做 EMA 平滑 + 极小容差（默认 0.02）
    """
    eps = 1e-6
    p_f = _softmax_stable(logits_f, eps)
    p_a = _softmax_stable(logits_a, eps)
    p_v = _softmax_stable(logits_v, eps)

    kl_f_a = _kl_pq(p_f, p_a, eps=eps).mean().item()
    kl_f_v = _kl_pq(p_f, p_v, eps=eps).mean().item()
    delta  = kl_f_v - kl_f_a  # >0 → visual 更偏离融合

    m = float(getattr(args, "decide_ema", 0.9))
    ema_prev = float(getattr(args, "_decide_ema_state", 0.0))
    ema = m * ema_prev + (1.0 - m) * delta
    args._decide_ema_state = ema

    eps_decide = float(getattr(args, "decide_eps", 0.02))
    if ema > eps_decide:
        corr = 'visual'
    elif ema < -eps_decide:
        corr = 'audio'
    else:
        corr = None

    if getattr(args, "debug_decide", False):
        print(f"[DECIDE] KLf|a={kl_f_a:.4f} KLf|v={kl_f_v:.4f} Δ={delta:.4f} EMA={ema:.4f} -> {corr}")
    return corr

def set_requires_grad_ln(model: nn.Module, modality: str):
    """
    冻结所有 LN，仅解冻指定模态的 LN（audio: *_a；visual: *_v）
    兼容 DataParallel：传入的是 DataParallel 包了一层的模型
    """
    core = model.module if hasattr(model, "module") else model

    # 先全部冻住
    # for n, m in core.named_modules():
    #     if isinstance(m, nn.LayerNorm):
    #         for pn, p in m.named_parameters(recurse=False):
    #             if pn in ("weight", "bias"):
    #                 p.requires_grad = False

    if modality is None:
        return

    # 只解冻坏模态 LN：名称里包含 *_a 或 *_v
    key = "_a" if modality == "audio" else "_v"
    for n, m in core.named_modules():
        if isinstance(m, nn.LayerNorm) and (("norm1" in n) or ("norm2" in n)) and (key in n):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias"):
                    p.requires_grad = True

# --------------------- 其余工具 ---------------------

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
    CE + REM + BAL + (一致性: 这里保留 PoE 版本示例；你可换回 MLA 双 KL)
    + 按批在线判定坏模态 → 仅解冻该模态 LayerNorm
    """
    # ---------- (1) Forward (adapt; keep graph) ----------
    with autocast():
        outputs, attn_raw, cached_av_feat = model.module.forward_eval(
            a=x[0], v=x[1],
            mode=args.testmode,
            args=args,
            stage='adapt',
            labels=labels,
            ret_av_feat=True,
            output_embedding=True
        )

    # ---------- (2) Unimodal anchors（用于一致性和腐败判定） ----------
    with torch.no_grad():
        logits_a_anchor, _ = model.module.forward_eval(a=x[0], v=None,  mode='audio',  args=args, stage='test')
        logits_v_anchor, _ = model.module.forward_eval(a=None,  v=x[1], mode='visual', args=args, stage='test')

    # ---------- (2.1) 判定被污染模态 & 动态解冻 LN ----------
    corrupted = decide_corrupted_modality(outputs, logits_a_anchor, logits_v_anchor, args)
    prev = getattr(args, "_last_corr", None)
    switch = (corrupted is not None) and (corrupted != prev)

    if switch:
        set_requires_grad_ln(model, corrupted)
        args._last_corr = corrupted
        if getattr(args, "debug_decide", False):
            print(f"[DECIDE] switch LN to train: {corrupted}")

        # ===== 关键：切换后重算一次前向，让 LN 进入图 =====
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs, attn_raw, cached_av_feat = model.module.forward_eval(
                a=x[0], v=x[1],
                mode=args.testmode, args=args,
                stage='adapt', labels=labels,
                ret_av_feat=True, output_embedding=True
            )
        with torch.no_grad():
            logits_a_anchor, _ = model.module.forward_eval(a=x[0], v=None,  mode='audio',  args=args, stage='test')
            logits_v_anchor, _ = model.module.forward_eval(a=None,  v=x[1], mode='visual', args=args, stage='test')


    # ---------- (3) MLA 权重（保持你原逻辑） ----------
    q_a, q_v = compute_modality_weights_mla(
        logits_a_anchor, logits_v_anchor, tau=getattr(args, 'mla_tau', 1.0)
    )
    gap_val = float((q_a - q_v).abs().mean().item())

    # ---------- (4) Core losses ----------
    pred = outputs.softmax(dim=-1)
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)
    C = p_sorted.size(1)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)

    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam1 = getattr(args, "re_lambda1", 0.0)
    lam_eff = max(1e-6, float(lam0 + lam1 * gap_val))
    w = torch.exp(-lam_eff * rank); w = w / (w.sum() + 1e-12)

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

    # ---------- (5) 一致性（PoE 版本；如需 MLA 双 KL，把这里换回去即可） ----------
    eps = 1e-6
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

    # ---------- (6) Total ----------
    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
            + loss_ra * getattr(args, "ent_weight", 0.0)
            - loss_bal * getattr(args, "bal_weight", 0.0)
            + loss_cons * getattr(args, "cons_weight", 0.2))

    # ---------- (7) Backprop / step ----------
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0
    )
    scaler.step(optimizer)
    scaler.update()

    # ---------- (8) Second forward (report) ----------
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

# --------------------- 只收集 LN 参数（audio/visual 两边都收） ---------------------

def collect_params(model):
    """
    收集 blocks_a / blocks_v 下 LayerNorm 的 weight/bias 参数（两边都收）。
    即使当前 requires_grad=False 也加入 optimizer，便于运行时动态切换。
    """
    params, names = [], []
    core = model.module if hasattr(model, "module") else model
    for nm, mod in core.named_modules():
        if isinstance(mod, nn.LayerNorm) and (("blocks_a" in nm) or ("blocks_v" in nm)):
            for pn, p in mod.named_parameters(recurse=False):
                if pn in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{pn}")
        if nm == 'module.blocks_u.0.attn.qkv' or nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    if len(params) == 0:
        print("[WARN] No LayerNorm params found under blocks_a/blocks_v.")
    else:
        print(f"[INFO] Collected LN params: {len(params)} tensors.")
    return params, names

# --------------------- 配置：初始全冻结，待判定后切换 LN ---------------------

def configure_model(model):
    model.train()
    for p in model.parameters():
        p.requires_grad = False

    for nm in model.module.blocks_u:
        for m in nm.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            m.requires_grad_(True)

    # 初始不解冻任何 LN；等 forward 第一次决策后再 set_requires_grad_ln 切换
    # 但 optim 里需要包含这些 LN，因此通过 collect_params 收集（即使现在是冻结的）
    print("[CONFIG] All params frozen initially. LN will be unfrozen after corruption decision at runtime.")
    return model




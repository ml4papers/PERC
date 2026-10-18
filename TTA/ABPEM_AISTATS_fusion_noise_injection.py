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

# ====== Fusion-input Gaussian noise (hook-based) ======
class _FusionNoiseHook:
    def __init__(self, std: float):
        self.std = float(std)
        self.handle = None

    def _pre_hook(self, module, inputs):
        # inputs: tuple(tensor, ...)
        x = inputs[0]
        if self.std <= 0.0:
            return inputs
        noise = torch.randn_like(x) * self.std
        return (x + noise, )

    def attach(self, target_module: nn.Module):
        self.handle = target_module.register_forward_pre_hook(self._pre_hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

from contextlib import contextmanager

@contextmanager
def fuse_noise(model: nn.Module, std: float):
    """
    仅对融合块入口（blocks_u.0.norm1）注入高斯噪声；退出时自动移除。
    """
    try:
        # DataParallel 兼容
        core = model.module if hasattr(model, "module") else model
        # 你给出的 CAV-MAE 结构里，融合层只有一个 U-block
        norm1 = core.module.blocks_u[0].norm1 if hasattr(core, "module") else core.blocks_u[0].norm1
        hk = _FusionNoiseHook(std)
        hk.attach(norm1)
        yield
    finally:
        hk.remove()


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
    ABPEM (with REM-style RE) + MLA/PoE consistency + Adaptive Rank Weighting + Fusion-Noise Consistency.
    - Returns: (outputs_adapt, outputs_test), (ce, ra, bal, cons, total)
    """
    # ---------- (1) Forward (adapt; keep graph) ----------
    # ---------- (1) Forward (adapt; keep graph) ----------
    # 推荐默认噪声强度：0.05~0.12（ImageNet-C/KS-50 level 5 通常 0.06~0.10 更稳）
    fuse_std = float(getattr(args, 'fuse_noise_std', 0.08))
    use_dn2  = bool(getattr(args, 'aug_noise', True)) and fuse_std > 0.0

    with autocast():
        # 先干净前向（得到融合入口缓存等）
        outputs_c, attn_raw, cached_av_feat = model.module.forward_eval(
            a=x[0], v=x[1],
            mode=args.testmode, args=args,
            stage='adapt', labels=labels,
            ret_av_feat=True, output_embedding=True
        )

        if use_dn2:
            # 噪声仅打到融合入口：对 blocks_u.0.norm1 的输入加噪
            with fuse_noise(model, fuse_std):
                outputs_n, _attn2 = model.module.forward_eval(
                    a=x[0], v=x[1],
                    mode=args.testmode, args=args,
                    stage='adapt', labels=labels,
                    ret_av_feat=False, output_embedding=False
                )
            # 在已得到 outputs_c (clean) & outputs_n (noisy) 的地方，用 EWA 替换等权平均
            # 假设你已有 logits: outputs_c（clean），outputs_n（noisy）
            # 用于融合: outputs = (1 - w_n) * outputs_c + w_n * outputs_n

            # with torch.cuda.amp.autocast(enabled=False):
            #     eps = 1e-6
            #     p_c = torch.softmax(outputs_c.detach().float(), dim=1).clamp_min(eps)  # [B,C]
            #     p_n = torch.softmax(outputs_n.detach().float(), dim=1).clamp_min(eps)
            #
            #     # 1) 分歧度：JS-divergence（数值更稳）
            #     m = 0.5 * (p_c + p_n)
            #     kl_c_m = torch.sum(p_c * (p_c.log() - m.log()), dim=1)  # KL(p_c || m)
            #     kl_n_m = torch.sum(p_n * (p_n.log() - m.log()), dim=1)  # KL(p_n || m)
            #     d = 0.5 * (kl_c_m + kl_n_m)  # [B], JS-div
            #
            #     # 2) 分位数归一化（小 batch 推荐用 20%-80% 分位）
            #     ql = torch.quantile(d, 0.20)
            #     qh = torch.quantile(d, 0.80)
            #     scale = (qh - ql).clamp_min(1e-6)
            #     d_norm = ((d - ql) / scale).clamp(0.0, 1.0)  # [0,1], 越大=分歧越大
            #
            #     # 3) 非线性压缩 → 噪声权重 w_n（分歧越大，w_n 越小）
            #     gamma = getattr(args, "wn_gamma", 3.0)  # 建议 2.0~4.0，越大越“果断”
            #     w_n = (1.0 - d_norm).pow(gamma).unsqueeze(1)  # [B,1]
            #
            #     # 可选：加入 clean 置信度进一步压小 w_n（clean 很自信时更信 clean）
            #     if getattr(args, "wn_conf_gate", True):
            #         conf_c = p_c.max(dim=1).values.unsqueeze(1)  # [B,1]
            #         w_n = w_n * (1.0 - conf_c) + w_n * 0.1  # 给个底座，避免完全为0
            #     # print('w_n', w_n)
            #
            # # 用更“果断”的 w_n 做线性融合（注意 logits 线性可加）
            # outputs = (1.0 - w_n.to(outputs_c.dtype)) * outputs_c + w_n.to(outputs_c.dtype) * outputs_n

            outputs = outputs_n
        else:
            outputs = outputs_c
    # # 推荐默认噪声强度：0.05~0.12（ImageNet-C/KS-50 level 5 通常 0.06~0.10 更稳）
    # fuse_std = float(getattr(args, 'fuse_noise_std', 0.08))
    # use_dn2  = bool(getattr(args, 'aug_noise', True)) and fuse_std > 0.0
    #
    # with autocast():
    #     # 先干净前向（得到融合入口缓存等）
    #     outputs_c, attn_raw, cached_av_feat = model.module.forward_eval(
    #         a=x[0], v=x[1],
    #         mode=args.testmode, args=args,
    #         stage='adapt', labels=labels,
    #         ret_av_feat=True, output_embedding=True
    #     )
    #
    #     if use_dn2:
    #         # 噪声仅打到融合入口：对 blocks_u.0.norm1 的输入加噪
    #         with fuse_noise(model, fuse_std):
    #             outputs_n, _attn2, _ = model.module.forward_eval(
    #                 a=x[0], v=x[1],
    #                 mode=args.testmode, args=args,
    #                 stage='adapt', labels=labels,
    #                 ret_av_feat=False, output_embedding=False
    #             )
    #         # log-space 平均更稳：logsumexp([clean, noisy]) - log(2)
    #         outputs = torch.logsumexp(torch.stack([outputs_c, outputs_n], dim=0), dim=0) - math.log(2.0)
    #     else:
    #         outputs = outputs_c

    # ---------- (2) MLA weights first (for adaptive rank + consistency) ----------
    with torch.no_grad():
        logits_a_anchor, _ = model.module.forward_eval(
            a=x[0], v=None, mode='audio',  args=args, stage='test'
        )
        logits_v_anchor, _ = model.module.forward_eval(
            a=None,  v=x[1], mode='visual', args=args, stage='test'
        )

    q_a, q_v = compute_modality_weights_mla(
        logits_a_anchor, logits_v_anchor, tau=getattr(args, 'mla_tau', 1.0)
    )  # [B,1] each
    # batch-level modality dominance gap
    gap_val = float((q_a - q_v).abs().mean().item())

    # ---------- (3) Core losses ----------
    # REM: Ranked Entropy with ADAPTIVE rank decay
    pred = outputs.softmax(dim=-1)                                  # [B, C]
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)         # [B, C]
    C = p_sorted.size(1)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)

    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam1 = getattr(args, "re_lambda1", 0.)   # default 0 keeps backward-compatibility
    lam_eff = max(1e-6, float(lam0 + lam1 * gap_val))               # scalar
    w = torch.exp(-lam_eff * rank); w = w / (w.sum() + 1e-12)       # [C]

    ent_sorted = -p_sorted * torch.log(p_sorted + 1e-6)
    loss_ra = (ent_sorted * w.unsqueeze(0)).sum(dim=-1).mean()

    # Balancing
    if getattr(args, "fix_loss_bal", False):
        p_sum = outputs.softmax(dim=-1).mean(dim=-2)
        loss_bal = - (p_sum * p_sum.log()).sum()
    else:
        p_sum = outputs.softmax(dim=-1).sum(dim=-2)
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()

    # ------------------Small pseudo-CE--------------------------
    with torch.no_grad():
        pseudo = outputs.argmax(dim=1)
    ce_loss = torch.nn.functional.cross_entropy(outputs, pseudo)

    # ---------- (4) Consistency（你原有的 MLA/PoE 任选其一，这里保持 PoE 示例） ----------
    eps = 1e-6
    p_f = pred  # fused probs, [B,C]
    p_a = torch.softmax(logits_a_anchor, dim=1).clamp_min(eps)  # [B,C]
    p_v = torch.softmax(logits_v_anchor, dim=1).clamp_min(eps)  # [B,C]

    with torch.no_grad():
        C = p_f.size(1)
        logC = math.log(C + 1e-12)
        Ha = -(p_a * p_a.log()).sum(dim=1) / logC
        Hv = -(p_v * p_v.log()).sum(dim=1) / logC
        ra = 1.0 / (Ha + 1e-6)
        rv = 1.0 / (Hv + 1e-6)
        gamma = (ra / (ra + rv)).unsqueeze(1)  # [B,1]

    log_t = gamma * p_a.log() + (1.0 - gamma) * p_v.log()
    t = torch.softmax(log_t, dim=1).clamp_min(eps)
    loss_cons = torch.sum(p_f * (p_f.add(eps).log() - t.log()), dim=1).mean()



    # ---------- (5) Total ----------
    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
            + loss_ra * getattr(args, "ent_weight", 0.0)
            - loss_bal * getattr(args, "bal_weight", 0.0)
            + loss_cons * getattr(args, "cons_weight", 0.2)
            + loss_fuse_cons * fuse_cons_w)

    # ---------- (6) Backprop / step ----------
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0
    )
    scaler.step(optimizer)
    scaler.update()

    # ---------- (7) Second forward (report) ----------
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


requires_grad_list = [
    'module.blocks_u.0.attn.qa',
    'module.blocks_u.0.attn.ka',
    'module.blocks_u.0.attn.va',
    'module.blocks_u.0.attn.qv',
    'module.blocks_u.0.attn.kv',
    'module.blocks_u.0.attn.vv'
]


def collect_params(model):
    params_fusion_qkv = []
    names_fusion_qkv = []
    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")
    return params_fusion_qkv, names_fusion_qkv


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            print(f"[INFO] Unfreezing module: {nm}")
            for param in m.parameters():
                param.requires_grad = True
    return model



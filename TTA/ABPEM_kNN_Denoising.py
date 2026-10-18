import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from collections import deque

# ===================== Fusion-level Memory-Bank Smoothing =====================

class FeatureBank:
    """
    跨 batch 的小型特征库：存高置信融合特征及其伪标签。
    - 仅保存 conf>=tau 的样本，容量上限 max_size（先进先出）
    - query_and_smooth: 用库里同类(伪标签)邻居做 kNN 平滑；不足时回退到 EMA 类原型
    """
    def __init__(self, num_classes: int, feat_dim: int,
                 max_size: int = 1024, tau: float = 0.7,
                 k: int = 5, beta: float = 0.2, proto_m: float = 0.9,
                 device: str = "cuda"):
        self.C = num_classes
        self.D = feat_dim
        self.max_size = max_size
        self.tau = tau
        self.k = k
        self.beta = beta
        self.dev = device

        self._feats = deque()
        self._labels = deque()
        self._confs = deque()

        self.protos = torch.zeros(self.C, self.D, device=device)
        self.proto_inited = torch.zeros(self.C, dtype=torch.bool, device=device)
        self.proto_m = proto_m

    @torch.no_grad()
    def _normalize(self, x):
        return torch.nn.functional.normalize(x, dim=1)

    @torch.no_grad()
    def add(self, feats: torch.Tensor, probs: torch.Tensor):
        """
        feats: [B,D] 融合特征（float32）
        probs: [B,C] softmax 概率
        """
        if feats is None or probs is None: return
        conf, pred = probs.max(dim=1)  # [B]
        mask = conf >= self.tau
        if mask.sum() == 0:
            return

        f = feats[mask].detach()
        y = pred[mask].detach()
        c = conf[mask].detach()

        for i in range(f.size(0)):
            self._feats.append(f[i].to(self.dev))
            self._labels.append(y[i].to(self.dev))
            self._confs.append(c[i].to(self.dev))
            if len(self._feats) > self.max_size:
                self._feats.popleft(); self._labels.popleft(); self._confs.popleft()

        # 更新类原型（单位向量 EMA）
        f_n = self._normalize(f)
        for cls in y.unique():
            ci = int(cls.item())
            mu = f_n[y == ci].mean(dim=0)  # [D]
            if self.proto_inited[ci]:
                self.protos[ci].mul_(self.proto_m).add_(mu, alpha=1.0 - self.proto_m)
            else:
                self.protos[ci] = mu
                self.proto_inited[ci] = True

    @torch.no_grad()
    def query_and_smooth(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        对当前 batch 的融合特征做 kNN 平滑（同类优先）。
        若库里同类邻居 < 2，则回退到原型；若仍无原型，则返回原feat。
        返回 [B,D]
        """
        if feats is None or feats.ndim != 2 or len(self._feats) == 0 or probs is None:
            return feats

        B, D = feats.size()
        x = self._normalize(feats)                  # [B,D]
        conf, pred = probs.max(dim=1)               # [B]

        F = torch.stack(list(self._feats))          # [N,D]
        Y = torch.stack(list(self._labels))         # [N]
        out = feats.clone()

        for i in range(B):
            cls = int(pred[i].item())
            idx_cls = (Y == cls).nonzero(as_tuple=False).squeeze(-1)
            if idx_cls.numel() >= 2:
                fbank = F.index_select(0, idx_cls)              # [Nc,D]
                fbank_n = self._normalize(fbank)
                s = (x[i:i+1] @ fbank_n.t()).squeeze(0)         # [Nc]
                k_use = min(self.k, s.numel())
                val, top = torch.topk(s, k=k_use, largest=True)
                w = torch.softmax(10.0 * val, dim=0)            # 温度固定10，减少新超参
                x_agg = (w.unsqueeze(1) * fbank_n.index_select(0, top)).sum(dim=0)  # [D]
                x_s = self._normalize((1.0 - self.beta) * x[i] + self.beta * x_agg.unsqueeze(0)).squeeze(0)
                out[i] = x_s * feats[i].norm().clamp_min(1e-6)
            else:
                # 回退到原型
                if self.proto_inited[cls]:
                    p = torch.nn.functional.normalize(self.protos[cls:cls+1], dim=1).squeeze(0)  # [D]
                    x_s = self._normalize((1.0 - self.beta) * x[i] + self.beta * p.unsqueeze(0)).squeeze(0)
                    out[i] = x_s * feats[i].norm().clamp_min(1e-6)
                else:
                    out[i] = feats[i]

        return out

# ===================== Utilities =====================

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

        # ====== 找到 head 维度，初始化 FeatureBank ======
        self.args._read_self = self  # 让函数能访问到 READ 实例
        head = self.model.module.mlp_head if hasattr(self.model, "module") else self.model.mlp_head
        if isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
            feat_dim = head[-1].in_features
            num_classes = head[-1].out_features
            self._clf_head = head
        elif isinstance(head, nn.Linear):
            feat_dim = head.in_features
            num_classes = head.out_features
            self._clf_head = head
        else:
            raise RuntimeError("Cannot infer classifier head for FeatureBank.")

        # 超参默认值（少量，可不传）
        fb_size   = int(getattr(self.args, "fb_size", 1024))
        fb_tau    = float(getattr(self.args, "fb_tau", 0.7))
        fb_k      = int(getattr(self.args, "fb_k", 5))
        fb_beta   = float(getattr(self.args, "fb_beta", 0.2))
        fb_pm     = float(getattr(self.args, "fb_proto_m", 0.9))
        self.fb = FeatureBank(num_classes, feat_dim, fb_size, fb_tau, fb_k, fb_beta, fb_pm, device)

        # 融合 logits 的线性融合系数（=0 即完全关闭该模块）
        self.fuse_dn_alpha = float(getattr(self.args, "fuse_dn_alpha", 0.3))

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
    ABPEM (with REM-style RE) + MLA-weighted consistency + Adaptive Rank Weighting
    + Fusion-level memory-bank kNN smoothing（不改损失，仅改前向的 logits）。
    """
    # ---------- (1) Forward (adapt; keep graph) ----------
    with autocast():
        outputs, attn_raw, cached_av_feat = model.module.forward_eval(
            a=x[0], v=x[1],
            mode=args.testmode,      # 'multimodal'
            args=args,
            stage='adapt',
            labels=labels,
            ret_av_feat=True,
            output_embedding=True
        )

    # ---------- (1.b) 融合层去噪 + logits 融合（小批友好） ----------
    alpha = float(getattr(args, "fuse_dn_alpha", getattr(args, "_read_self").fuse_dn_alpha if hasattr(args, "_read_self") else 0.3))
    if alpha > 0.0 and cached_av_feat is not None:
        with torch.no_grad():
            p_f_raw = torch.softmax(outputs.detach().to(torch.float32), dim=-1)  # [B,C]
            z_dn = args._read_self.fb.query_and_smooth(cached_av_feat.detach().to(torch.float32), p_f_raw)  # [B,D]
        # 只过 head，得到去噪 logits
        with torch.cuda.amp.autocast(enabled=False):
            logits_dn = args._read_self._clf_head(z_dn)  # [B,C], fp32 更稳
        # 与原 logits 融合（不改变你的损失定义）
        outputs = (1.0 - alpha) * outputs + alpha * logits_dn.to(outputs.dtype)

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

    # ---------- (4) MLA-weighted consistency ----------
    eps = 1e-6
    p_f = pred
    p_a = torch.softmax(logits_a_anchor, dim=1)
    p_v = torch.softmax(logits_v_anchor, dim=1)

    if getattr(args, "cons_conf_th", 0.0) > 0.0:
        conf = p_f.max(dim=1).values
        mask = (conf >= args.cons_conf_th).float().unsqueeze(1)  # [B,1]
    else:
        mask = 1.0

    log_pf = torch.log(p_f + eps)
    kl_f_a = (p_f * (log_pf - torch.log(p_a + eps))).sum(dim=1, keepdim=True)  # [B,1]
    kl_f_v = (p_f * (log_pf - torch.log(p_v + eps))).sum(dim=1, keepdim=True)  # [B,1]
    loss_cons = ((q_a * mask * kl_f_a).mean() + (q_v * mask * kl_f_v).mean())

    # ---------- (4.b) 更新特征库（用融合后的 p_f） ----------
    with torch.no_grad():
        args._read_self.fb.add(cached_av_feat.detach().to(torch.float32), p_f.detach().to(torch.float32))

    # ---------- (5) Total ----------
    loss = (ce_loss * getattr(args, "ce_weight", 1.0)
            + loss_ra * getattr(args, "ent_weight", 0.0)
            - loss_bal * getattr(args, "bal_weight", 0.0)
            + loss_cons * getattr(args, "cons_weight", 1.0))

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

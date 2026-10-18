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
    H_a = -(p_a * (torch.log(p_a + eps))).sum(dim=1)
    H_v = -(p_v * (torch.log(p_v + eps))).sum(dim=1)
    H = torch.stack([H_a, H_v], dim=1)
    H_shift = H - H.max(dim=1, keepdim=True).values
    w = torch.softmax(-H_shift / max(tau, 1e-6), dim=1)
    q_a = w[:, 0:1]
    q_v = w[:, 1:2]
    return q_a, q_v

# ================= AdaptFormer adapter =================
# ===== AdaptFormer-lite (parallel residual) =====
import math
import torch
import torch.nn as nn

class AdaptMLP(nn.Module):
    """并联残差适配器：y <- y + beta * Up(Drop(LN(y)) Down)"""
    def __init__(self, dim: int, r: int = 4, beta: float = 0.5, p: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        # 为了稳定，默认不学仿射
        self.ln.elementwise_affine = False

        self.down = nn.Linear(dim, r, bias=False)
        self.up   = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

        self.beta = float(beta)
        self.drop = nn.Dropout(p) if p and p > 1e-8 else nn.Identity()

    def forward(self, y: torch.Tensor):
        z = self.down(self.drop(self.ln(y)))
        z = self.up(z)
        return y + self.beta * z

class BlockWithAdapter(nn.Module):
    """包装原融合 Block：先跑原 block，再并联注入小适配器"""
    def __init__(self, block: nn.Module, dim: int, r=4, beta=0.5, p=0.0):
        super().__init__()
        self.block = block
        self.adapter = AdaptMLP(dim, r=r, beta=beta, p=p)

    def forward(self, x, *args, **kwargs):
        # 兼容你现有的 Block 返回 (y, attn)
        y, attn = self.block(x, *args, **kwargs)
        y = self.adapter(y)
        return y, attn

def _parse_device_dtype(module: nn.Module):
    # 找一个权重拿到 device/dtype
    for p in module.parameters():
        return p.device, p.dtype
    return torch.device("cpu"), torch.float32

def attach_adaptformer_to_fusion(model: nn.Module,
                                 fusion_block_names=("blocks_u.0",),
                                 dim: int = 768,
                                 r: int = 4,
                                 beta: float = 0.5,
                                 dropout: float = 0.0):
    """
    仅在融合层 block（默认 blocks_u.0）外包一层并联适配器
    """
    named = dict(model.named_modules())
    added_params = 0
    replaced = 0

    for name, mod in list(model.named_modules()):
        # 精确匹配融合 block 名称
        if any(name.endswith(tgt) or name == tgt for tgt in fusion_block_names):
            # name 对应的对象是 Block；找到其 parent
            if '.' in name:
                parent_name, leaf = name.rsplit('.', 1)
                parent = named[parent_name]
            else:
                parent = model
                leaf = name

            dev, dtyp = _parse_device_dtype(mod)
            wrapper = BlockWithAdapter(mod, dim=dim, r=r, beta=beta, p=dropout)
            wrapper = wrapper.to(device=dev, dtype=dtyp)
            setattr(parent, leaf, wrapper)

            # 统计新增参数（只算 adapter 的 down/up）
            added_params += sum(p.numel() for n, p in wrapper.named_parameters() if "adapter." in n and p.requires_grad)
            replaced += 1

    return replaced, added_params

# ======= 只开启融合层 Adapter 的可训练参数 =======
def collect_params(model: nn.Module):
    adapt_params, names = [], []
    for n, p in model.named_parameters():
        if "adapter.down" in n or "adapter.up" in n:
            p.requires_grad_(True)
            adapt_params.append(p)
            names.append(n)
        else:
            # 其余一律冻结（含 QKV、FFN、LN、head 等）
            p.requires_grad_(False)
    if not adapt_params:
        print("[AdaptFormer] No adapter parameters found!")
    else:
        print(f"[AdaptFormer] Trainable adapter tensors: {len(adapt_params)}")
    return adapt_params, names


def configure_model(model: nn.Module):
    """
    只对融合层（blocks_u.0）插入并联适配器；其它全部冻结
    从 model.args 读取：use_adaptformer, adapt_r, adapt_beta, adapt_dropout
    """
    args = getattr(model, "args", None)
    use_af = bool(getattr(args, "use_adaptformer", True)) if args is not None else True
    r      = int(getattr(args, "adapt_r", 4)) if args is not None else 4
    beta   = float(getattr(args, "adapt_beta", 0.5)) if args is not None else 0.5
    p      = float(getattr(args, "adapt_dropout", 0.0)) if args is not None else 0.0

    model.train()
    for p_all in model.parameters():
        p_all.requires_grad = False

    if use_af:
        replaced, added = attach_adaptformer_to_fusion(
            model,
            fusion_block_names=("blocks_u.0",),
            dim=768, r=r, beta=beta, dropout=p
        )
        print(f"[AdaptFormer] use_adaptformer=True | replaced_blocks={replaced} | added_params={added}")
    else:
        print("[AdaptFormer] disabled; model unchanged.")

    return model

# ===================== Core TTA =====================

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
                loss_tuple = (torch.tensor(0.0, device=x[0].device),) * 5
            return outputs, loss_tuple

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    # (1) forward adapt
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

    # (2) MLA anchors
    with torch.no_grad():
        logits_a_anchor, _ = model.module.forward_eval(a=x[0], v=None, mode='audio',  args=args, stage='test')
        logits_v_anchor, _ = model.module.forward_eval(a=None,  v=x[1], mode='visual', args=args, stage='test')

    q_a, q_v = compute_modality_weights_mla(
        logits_a_anchor, logits_v_anchor, tau=getattr(args, 'mla_tau', 1.0)
    )
    gap_val = float((q_a - q_v).abs().mean().item())

    # (3) losses: REM / BAL / CE
    pred = outputs.softmax(dim=-1)
    p_sorted, _ = torch.sort(pred, dim=-1, descending=True)
    C = p_sorted.size(1)
    rank = torch.arange(C, device=pred.device, dtype=pred.dtype)

    lam0 = getattr(args, "re_lambda0", getattr(args, "re_lambda", 0.0))
    lam1 = getattr(args, "re_lambda1", 0.)
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

    # (4) MLA consistency
    eps = 1e-6
    p_f = pred
    p_a = torch.softmax(logits_a_anchor, dim=1)
    p_v = torch.softmax(logits_v_anchor, dim=1)

    if args.cons_conf_th > 0.0:
        conf = p_f.max(dim=1).values
        mask = (conf >= args.cons_conf_th).float().unsqueeze(1)
    else:
        mask = 1.0

    log_pf = torch.log(p_f + eps)
    kl_f_a = (p_f * (log_pf - torch.log(p_a + eps))).sum(dim=1, keepdim=True)
    kl_f_v = (p_f * (log_pf - torch.log(p_v + eps))).sum(dim=1, keepdim=True)
    loss_cons = ((q_a * mask * kl_f_a).mean() + (q_v * mask * kl_f_v).mean())

    loss = (ce_loss * args.ce_weight
            + loss_ra * args.ent_weight
            - loss_bal * args.bal_weight
            + loss_cons * args.cons_weight)

    # (5) step
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0
    )
    scaler.step(optimizer)
    scaler.update()

    # (6) report
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





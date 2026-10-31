import torch
from torch import nn as nn

import math
import torch
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    k = 2 * r + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=r)


def _annotation_entropy(gt_all: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    gt_all: (B,R,1,H,W), binary masks.
    returns normalized entropy map: (B,1,H,W), values in [0,1].
    """
    p = gt_all.float().mean(dim=1)  # (B,1,H,W)

    h = -(p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps))
    h = h / math.log(2.0)

    return h.clamp(0.0, 1.0)


def information_weighted_dou_loss(
    pred_logits: torch.Tensor,   # (B,1,1,H,W) or (B,1,H,W)
    gt_all: torch.Tensor,        # (B,R,1,H,W)
    rater_idx: int = 0,
    tau: int = 2,
    lam: float = 3.0,
    alpha: float = 0.5,
    rho: float = 1.0,
    delta: float = 0.0,
    normalize_weight: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Information-Weighted DABL.

    Difference from the current DABL:
      - Instead of binary disagreement D = union - intersection,
        use annotation entropy H(x).
      - Optional rater-specific deviation term controlled by delta.
    """
    if pred_logits.dim() == 5:
        pred_logits = pred_logits.squeeze(2)  # (B,1,H,W)

    P = torch.sigmoid(pred_logits)
    G = gt_all[:, rater_idx]  # (B,1,H,W)

    # 1) information content of annotation distribution
    H = _annotation_entropy(gt_all, eps=eps)      # (B,1,H,W)
    H_tau = _dilate(H, r=tau).clamp(0.0, 1.0)     # ambiguity band

    # 2) controllable sharpness
    H_tau = H_tau.pow(rho)

    # 3) optional rater-specific deviation from group tendency
    if delta > 0:
        p = gt_all.float().mean(dim=1)            # (B,1,H,W)
        M_r = torch.abs(G - p)                    # (B,1,H,W)
        M_tau = _dilate(M_r, r=tau).clamp(0.0, 1.0)
        info_weight = H_tau * (1.0 + delta * M_tau)
    else:
        info_weight = H_tau

    # 4) final spatial weights
    w = 1.0 + lam * info_weight

    # Optional: stabilize across images with different disagreement amount
    if normalize_weight:
        w = w / (w.mean(dim=(1, 2, 3), keepdim=True) + eps)

    # 5) weighted soft DoU
    Iw = (w * P * G).sum(dim=(1, 2, 3))
    Uw = (w * (P + G - P * G)).sum(dim=(1, 2, 3))

    loss = (Uw - Iw) / (Uw - alpha * Iw + eps)
    return loss.mean()
    
class BoundaryFocusedDoULoss(nn.Module):
    """
    Wraps information_weighted_dou_loss for the stage-2 training loop.

    The training loop calls loss_fct(y_preds, masks) where both tensors have
    shape (B, E, H, W) — one channel per expert.  This class:
      1. Iterates over all E experts.
      2. Adds the spatial channel dim expected by information_weighted_dou_loss:
            masks (B,E,H,W) → gt_all (B,E,1,H,W).
      3. Returns the mean loss across experts.

    Args:
        tau:              Dilation radius for the ambiguity band (pixels).
        lam:              Weight of the entropy-based focus term.
        alpha:            DoU denominator interpolation (0 = IoU, 0.5 ≈ Dice).
        rho:              Sharpness exponent on the entropy map.
        delta:            If > 0, adds rater-specific deviation weighting.
        normalize_weight: Normalise spatial weights per image (improves stability).
    """

    def __init__(
        self,
        tau: int = 2,
        lam: float = 3.0,
        alpha: float = 0.5,
        rho: float = 1.0,
        delta: float = 0.0,
        normalize_weight: bool = True,
    ):
        super().__init__()
        self.tau = tau
        self.lam = lam
        self.alpha = alpha
        self.rho = rho
        self.delta = delta
        self.normalize_weight = normalize_weight

    def forward(self, pred_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        pred_logits : (B, E, H, W)  raw logits, one per expert
        masks       : (B, E, H, W)  binary GT masks, one per expert
        """
        E = pred_logits.shape[1]
        gt_all = masks.unsqueeze(2)          # (B, E, 1, H, W)

        total = sum(
            information_weighted_dou_loss(
                pred_logits[:, i:i+1],       # (B, 1, H, W)
                gt_all,
                rater_idx=i,
                tau=self.tau,
                lam=self.lam,
                alpha=self.alpha,
                rho=self.rho,
                delta=self.delta,
                normalize_weight=self.normalize_weight,
            )
            for i in range(E)
        )
        return total / E


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        num_classes = input.shape[1]
        # target = torch.movedim(F.one_hot(target, num_classes), -1, 1)
        # compute per channel Dice coefficient
        # target = torch.movedim(target, -1, 1)
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect / denominator)

def init_optimization(model, args):

    learning_rate = 0.0001

    if args.model_name == 'pionono':
        opt_params = [
            {'params': model.unet.parameters()},
            {'params': model.head.parameters()},
            {'params': model.z.parameters(), 'lr': 0.02}
        ]
    elif 'cm' in args.model_name:
        opt_params = [
            {'params': model.seg_model.parameters()},
            {'params': model.cm_head.parameters(), 'lr': 0.01}
        ]
    elif args.model_name == 'PIRA' and args.stage == 1:
        opt_params = [
            {'params': model.unet.parameters()},
            {'params': model.prior.parameters()},
            {'params': model.posterior.parameters()},
            {'params': model.fcomb.parameters()}
        ]
    elif args.model_name == 'PIRA' and args.stage == 2:
        opt_params = [
            {'params': model.proj_heads.parameters()}
        ]
    elif args.model_name == 'prob_unet':
        opt_params = [
            {'params': model.parameters()}
        ]

    optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

    if getattr(args, 'model_name', '') == 'PIRA' and getattr(args, 'stage', 1) == 2:
        loss_fct = BoundaryFocusedDoULoss(
            tau=getattr(args, 'loss_tau', 2),
            lam=getattr(args, 'loss_lam', 3.0),
            alpha=getattr(args, 'loss_alpha', 0.5),
            rho=getattr(args, 'loss_rho', 1.0),
            delta=getattr(args, 'loss_delta', 0.0),
        )
    else:
        loss_fct = GeneralizedDiceLoss(normalization='sigmoid')
    return optimizer, loss_fct
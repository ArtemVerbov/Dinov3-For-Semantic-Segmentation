import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import CrossEntropyLoss


def loss_fn(logits, labels, cross_entropy_coef: float, dice_mode: str, dice_logits: bool) -> torch.Tensor:
    cross_entropy_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(mode=dice_mode, from_logits=dice_logits)
    return cross_entropy_coef * cross_entropy_loss(logits, labels) + (1 - cross_entropy_coef) * dice_loss(logits, labels)

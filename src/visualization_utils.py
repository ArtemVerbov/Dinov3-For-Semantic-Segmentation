import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks, make_grid

from src.transforms import inverse_normalization


def visualize_mask(images: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:

    images_with_masks = []
    if len(logits.shape) == 4:
        predicted_masks = torch.argmax(logits, dim=1).cpu()
        num_classes = logits.shape[1]
        one_hot = torch.eye(num_classes, device='cpu')[predicted_masks]
        binary_masks = one_hot.permute(0, 3, 1, 2).bool()
        foreground_masks = binary_masks[:, 1:]
    else:
        num_classes = len(logits.unique())
        one_hot = F.one_hot(logits.long().cpu(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2)
        foreground_masks = one_hot[:, 1:].bool()

    for batch_idx in range(images.shape[0]):
        image = inverse_normalization(images[batch_idx].cpu())

        masks = foreground_masks[batch_idx]

        if image.dtype != torch.uint8:
            image = (image.clamp(0, 1) * 255).byte()

        image_with_mask = draw_segmentation_masks(
            image=image,
            masks=masks,
            alpha=0.3,
            colors=["blue", "orange"]
        )
        images_with_masks.append(image_with_mask)

    return make_grid(images_with_masks, nrow=2)

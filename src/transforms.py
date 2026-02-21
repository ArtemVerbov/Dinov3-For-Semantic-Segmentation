import albumentations as albu
import torch
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size, transforms_type):
    if transforms_type in ['val', 'test']:
        transforms = albu.Compose(
            [
                albu.Resize(height=img_size, width=img_size),
                albu.Normalize(),  # Normalization
                ToTensorV2(),
            ],
        )
    else:
        transforms = albu.Compose(
            [
                albu.Resize(height=img_size, width=img_size),
                albu.HorizontalFlip(p=0.5),
                albu.Affine(
                    scale=(0.9, 1.1),
                    rotate=(-5, 5),
                    p=0.5
                ),
                albu.Normalize(),  # Normalization
                ToTensorV2(),
            ],
        )
    return transforms




def inverse_normalization(img, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return std * img + mean
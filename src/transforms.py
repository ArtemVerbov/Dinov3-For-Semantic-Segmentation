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
                albu.OneOf([
                    albu.Resize(height=img_size, width=img_size),
                ],
                    p=1.0
                ),
                albu.OneOf(
                    [
                        albu.RGBShift(p=1.0),
                        albu.HueSaturationValue(p=1.0),
                        albu.ChannelShuffle(p=1.0),
                        albu.CLAHE(p=1.0),
                        albu.RandomBrightnessContrast(p=1.0),
                        albu.RandomGamma(p=1.0),
                        albu.ToGray(p=1.0),
                    ],
                    p=1.0
                ),
                albu.OneOf(
                    [
                        albu.Blur(p=1.0),
                        albu.MedianBlur(p=1.0),
                        albu.GaussNoise(p=1.0),
                        albu.SaltAndPepper(p=1.0),
                        albu.ImageCompression(quality_range=(75, 85), p=1.0),
                    ],
                    p=1.0
                ),
                albu.HorizontalFlip(p=0.5),
                albu.Affine(
                    scale=(0.9, 1.1),
                    rotate=(-3, 3),
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
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from torch import nn
from transformers import DINOv3ConvNextBackbone


class FPNBackboneDINOv3(nn.Module):
    def __init__(
            self,
            dinov3_model_name: str,
            dino_out_indices: list[int],
            freeze_backbone: bool = True,
            encoder_depth: int = 3,
            encoder_pyramid_channels: int = 256,
            encoder_segmentation_channels: int = 128,
            number_of_classes: int = 1,
            head_activation: str = None,
            head_kernel_size: int = 1,
            head_upsampling: int = 4,
    ) -> None:
        super().__init__()
        self.dino_backbone = DINOv3ConvNextBackbone.from_pretrained(
            pretrained_model_name_or_path=dinov3_model_name,
            out_indices=dino_out_indices,
        )
        if freeze_backbone:
            for param in self.dino_backbone.parameters():
                param.requires_grad = False

        self.fpn_decoder = FPNDecoder(
            encoder_channels=self.dino_backbone.channels,
            encoder_depth=encoder_depth,
            pyramid_channels=encoder_pyramid_channels,
            segmentation_channels=encoder_segmentation_channels,
        )

        self.seg_head = SegmentationHead(
            in_channels=self.fpn_decoder.out_channels,
            out_channels=number_of_classes,
            activation=head_activation,
            kernel_size=head_kernel_size,
            upsampling=head_upsampling,
        )

    def forward(self, image):
        features = self.dino_backbone(image).feature_maps
        logits = self.fpn_decoder(features)
        return self.seg_head(logits)
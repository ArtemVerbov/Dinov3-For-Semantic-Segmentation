from functools import partial
from typing import TYPE_CHECKING

from lightning import LightningModule
from segmentation_models_pytorch.losses import DiceLoss, MULTICLASS_MODE
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.metrcis import get_metrics
from src.visualization_utils import visualize_mask

if TYPE_CHECKING:
    pass


class SegmentationLightningModule(LightningModule):  # noqa: WPS214
    def __init__(
        self,
        mask_to_labes: dict[str, int],
        model,
        optimizer: Optimizer | partial | None = None,
        module_cfg=None,
        scheduler: LRScheduler | None = None,
    ):
        super().__init__()

        metrics = get_metrics(
            num_classes=len(mask_to_labes),
            input_format='index',
            include_background=module_cfg.include_background,
        )

        self.loss = DiceLoss(MULTICLASS_MODE, from_logits=True)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.module_cfg = module_cfg
        self.model = model

        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters(ignore=['model'])

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: Tensor, batch_idx):  # noqa: WPS210

        images, targets = batch
        logits = self.forward(images)
        targets = targets.long()
        loss = self.loss(logits, targets)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'preds': logits, 'target': targets}

    def validation_step(self, batch: list[Tensor], batch_index: int):  # noqa: WPS210
        images, targets = batch
        logits = self.forward(images)

        pred_mask = logits.softmax(dim=1).argmax(dim=1)
        masks = targets.long()
        loss = self.loss(logits, masks)

        self._valid_metrics(pred_mask, masks)

        if batch_index <= 5:
            self._visualize(images, logits, batch_index)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self._valid_metrics(pred_mask, masks), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: list[Tensor], batch_idx: int):
        images, targets = batch
        logits = self.forward(images)

        pred_mask = logits.softmax(dim=1).argmax(dim=1)
        masks = targets.long()

        self._test_metrics(pred_mask, masks)
        self.log_dict(self._test_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # noinspection PyCallingNonCallable
    def configure_optimizers(self) -> dict:
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler:
            scheduler = self.scheduler(optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': self.module_cfg.interval,
                    'frequency': self.module_cfg.frequency,
                    'monitor': self.module_cfg.monitor,
                },
            }
        return {'optimizer': optimizer}

    def _visualize(self, images, logits, idx):

        grid = visualize_mask(images, logits)

        logger = self.logger.experiment
        logger.add_image(
            f'segmentation_batch_{idx}', # Use a more descriptive name
            grid,
            self.current_epoch,
        )

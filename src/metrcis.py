from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore, MeanIoU


def get_metrics(miou_per_class=False, dice_average='micro', **kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'mIoU': MeanIoU(per_class=miou_per_class, **kwargs),
            'dice_score': DiceScore(average=dice_average, **kwargs),
        },
    )

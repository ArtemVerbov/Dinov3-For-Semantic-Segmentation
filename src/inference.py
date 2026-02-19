from pathlib import Path

import cv2
import hydra
from clearml import Task
from lightning import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.configs import InferenceConfig
from src.constants import DATASETS_PATH, CONFIGS_DIR, ASSETS_PATH
from src.dataset import InferenceDataset
from src.lighting_module import SegmentationLightningModule
from src.model import FPNBackboneDINOv3
from src.transforms import get_transforms
from src.visualization_utils import visualize_mask


def get_model_weights(
    project_name: str = 'dinov3_fpn',
    task_name: str = 'semantic_segmentation-2026-01-26 13:40',
    model_name: str = 'best_model'
) -> Path:
    task = Task.get_task(project_name=project_name, task_name=task_name)
    local_path = task.artifacts[model_name].get_local_copy()
    return local_path


@hydra.main(config_path=str(CONFIGS_DIR), config_name='inference', version_base='1.2')
def inference(cfg: InferenceConfig):

    save_path = ASSETS_PATH / cfg.data_conf.dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = get_model_weights(
        project_name=cfg.project_conf.project_name,
        task_name=cfg.task_name,
        model_name=cfg.model_name,
    )

    dataset = InferenceDataset(
        DATASETS_PATH / cfg.data_conf.dataset_name,
        transforms=get_transforms(cfg.data_conf.img_size, 'test'),
        debug=cfg.data_conf.debug,
    )
    dataloader = DataLoader(
        dataset,
        num_workers=cfg.data_conf.num_workers,
        batch_size=cfg.data_conf.batch_size,
        shuffle=False,
    )
    model_conf_dict = OmegaConf.to_container(cfg.model_conf, resolve=True)

    lm = SegmentationLightningModule(
        model=FPNBackboneDINOv3(**model_conf_dict)
    )

    trainer_lm = Trainer()

    predicted = trainer_lm.predict(model=lm, ckpt_path=model_path, weights_only=False, dataloaders=dataloader)
    for pred_index, (image, mask_logits) in enumerate(predicted):
        masks = visualize_mask(image, mask_logits, inverse_normalize=False, alpha=0.2)
        np_image = masks.permute(1, 2, 0).cpu().numpy()

        cv2.imwrite(
            save_path / f'{pred_index}_img.jpeg',
            cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR),
        )





if __name__ == "__main__":
    inference()
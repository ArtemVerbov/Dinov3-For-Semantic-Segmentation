import datetime
import os

import hydra
from clearml import Task
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader

from src.configs import Config
from src.constants import DATASETS_PATH, CONFIGS_DIR
from src.lighting_module import SegmentationLightningModule
from src.model import FPNBackboneDINOv3
from src.dataset import OxfordPetBoundaryDataset
from src.transforms import get_transforms


@hydra.main(config_path=str(CONFIGS_DIR), config_name='train', version_base='1.2')
def main(cfg: Config):
    seed_everything(42)

    SimpleOxfordPetDataset.download(DATASETS_PATH / cfg.data_conf.dataset_name)

    debug = cfg.data_conf.debug
    if debug:
        print(f"Running debug mode, with {debug} number of samples")

    train_dataset = OxfordPetBoundaryDataset(
        root_dir=DATASETS_PATH / cfg.data_conf.dataset_name,
        transform=get_transforms(cfg.data_conf.img_size, 'train'),
        split='train',
        debug=debug,
    )

    val_dataset = OxfordPetBoundaryDataset(
        root_dir=DATASETS_PATH / cfg.data_conf.dataset_name,
        transform=get_transforms(cfg.data_conf.img_size, 'val'),
        split='val',
    )

    test_dataset = OxfordPetBoundaryDataset(
        root_dir=DATASETS_PATH / cfg.data_conf.dataset_name,
        transform=get_transforms(cfg.data_conf.img_size, 'test'),
        split='test',

    )

    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.data_conf.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        persistent_workers=cfg.data_conf.persistent_workers,
        pin_memory=cfg.data_conf.pin_memory,
    )

    val_dl = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.data_conf.batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        persistent_workers=cfg.data_conf.persistent_workers,
        pin_memory=cfg.data_conf.pin_memory,
    )

    test_dl = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.data_conf.batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        persistent_workers=cfg.data_conf.persistent_workers,
        pin_memory=cfg.data_conf.pin_memory,
    )

    mask_to_labes = train_dataset.mask_to_label

    cfg.model_conf.number_of_classes = len(mask_to_labes)
    model_conf_dict = OmegaConf.to_container(cfg.model_conf, resolve=True)
    training_conf = OmegaConf.to_container(cfg.trainer_conf, resolve=True)
    ckpt_conf = OmegaConf.to_container(cfg.model_ckpt_conf, resolve=True)

    optimizer_partial = hydra.utils.instantiate(cfg.optimizer)
    scheduler_partial = hydra.utils.instantiate(cfg.scheduler)

    if cfg.project_conf.track_in_clearml:
        Task.force_requirements_env_freeze()
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        task = Task.init(
            project_name=cfg.project_conf.project_name,
            task_name=f'{cfg.project_conf.experiment_name}-{date}',
            output_uri=True,
        )
        task.connect(cfg)
        task.connect_configuration(train_dataset.transform.transforms, name='transformations')

    fpnd = FPNBackboneDINOv3(
        **model_conf_dict
    )

    lm = SegmentationLightningModule(
        mask_to_labes=mask_to_labes,
        model=fpnd,
        optimizer=optimizer_partial,
        scheduler=scheduler_partial,
        module_cfg=cfg.module_conf,
    )

    model_checkpoint = None
    if cfg.trainer_conf.enable_checkpointing:
        model_checkpoint = ModelCheckpoint(
            **ckpt_conf
        )
    trainer_lm = Trainer(
        **training_conf,
        callbacks=model_checkpoint
    )

    trainer_lm.fit(
        model=lm,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )
    trainer_lm.test(dataloaders=test_dl)


if __name__ == "__main__":
    main()

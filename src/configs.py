from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.Adam"
    _partial_: bool = True
    lr: float = 2e-4

@dataclass
class SchedulerConfig:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    _partial_: bool = True
    patience: int = 3

@dataclass
class ModelConf:
    dinov3_model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
    dino_out_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    freeze_backbone: bool = True
    encoder_depth: int = 3
    encoder_pyramid_channels: int = 256
    encoder_segmentation_channels: int = 128
    number_of_classes: int = 1
    head_activation: Optional[str] = None
    head_kernel_size: int = 1
    head_upsampling: int = 4

@dataclass
class DataConfig:
    dataset_name: str = 'pets'
    debug: None | int = None
    img_size: int = 256
    batch_size: int = 1
    pin_memory: bool = True
    persistent_workers: bool = True

@dataclass
class TrainerConfig:
    max_epochs: int = 60
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 5
    precision: str = '32-true'
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = 1
    deterministic: bool = False
    fast_dev_run: bool = False
    accelerator: str = 'auto'
    enable_checkpointing: bool = False

@dataclass
class ModuleConfig:
    interval: str = 'epoch'
    frequency: int = 1
    monitor: str = 'val_loss'
    include_background: bool = False
    batches_to_visualize: int = 5

@dataclass
class ModelCkptConfig:
    monitor: str = 'val_loss_epoch'
    save_on_exception: bool = True
    mode: str = 'min'
    save_top_k: int = 1

@dataclass
class ProjectConfig:
    project_name: str = 'dinov3_fpn'
    experiment_name: str = 'semantic_segmentation'
    track_in_clearml: bool = True

@dataclass
class Config:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    model_conf: ModelConf = field(default_factory=ModelConf)
    data_conf: DataConfig = field(default_factory=DataConfig)
    trainer_conf: TrainerConfig = field(default_factory=TrainerConfig)
    module_conf: ModuleConfig = field(default_factory=ModuleConfig)
    model_ckpt_conf: ModelCkptConfig = field(default_factory=ModelCkptConfig)
    project_conf: ProjectConfig = field(default_factory=ProjectConfig)

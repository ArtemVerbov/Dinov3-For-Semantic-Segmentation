import os
from pathlib import Path

SEGMENTATION_IMAGE_SIZE = 640

_DEFAULT_PROJECT_PATH = Path(__file__).resolve().parents[1]

PROJECT_ROOT = Path(os.getenv('PROJ_ROOT', _DEFAULT_PROJECT_PATH))

DATASETS_PATH = os.getenv('DATASETS_PATH', PROJECT_ROOT / 'datasets')

ASSETS_PATH = os.getenv('ASSETS_PATH', PROJECT_ROOT / 'assets')

TRAIN_DIR = Path(os.getenv('TRAIN_DIR', PROJECT_ROOT / 'model_weights'))

CONFIGS_DIR = Path(os.getenv('CONFIGS_DIR', PROJECT_ROOT / 'configs'))

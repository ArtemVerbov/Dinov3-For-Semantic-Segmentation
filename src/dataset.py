import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.constants import DATASETS_PATH
from src.transforms import get_transforms
from src.visualization_utils import visualize_mask


class OxfordPetBoundaryDataset(Dataset):
    """
    A custom Dataset class for the Oxford-IIIT Pet Dataset that creates a 3-class
    segmentation mask: Background (0), Pet (1), and Boundary (2).

    This class now automatically handles the creation of 'train.txt' and 'val.txt'
    from 'trainval.txt' in a reproducible way.
    """

    def __init__(self, root_dir, split='train', transform=None, debug=None, train_val_split_ratio=0.9):
        """
        Args:
            root_dir (string): The root directory of the dataset.
            split (str): The dataset split, can be 'train', 'val', or 'test'.
            transform (callable, optional): An albumentations transform pipeline.
            debug (int, optional): If set, limits the dataset to the first `debug` samples.
            train_val_split_ratio (float): The ratio for the train/val split (e.g., 0.9 for 90% train).
                                           Only used when the split is first created.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'annotations', 'trimaps')

        if split in ['train', 'val']:
            self._create_train_val_split_if_needed(train_val_split_ratio)
        elif split != 'test':
            raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'val', or 'test'.")

        split_filename = f"{split}.txt"
        split_file_path = os.path.join(root_dir, 'annotations', split_filename)

        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"The split file '{split_filename}' was not found in {os.path.join(root_dir, 'annotations')}.")

        with open(split_file_path, 'r') as f:
            base_names = [line.strip().split(' ')[0] for line in f]

        self.image_files = [f"{name}.jpg" for name in base_names]

        if debug:
            self.image_files = self.image_files[:debug]

        self._mask_to_label = None

    def _create_train_val_split_if_needed(self, split_ratio):
        """
        Checks for and creates train.txt and val.txt from trainval.txt if they don't exist.
        This operation is idempotent: it only runs once.
        """
        annotations_dir = os.path.join(self.root_dir, 'annotations')
        trainval_path = os.path.join(annotations_dir, 'trainval.txt')
        train_path = os.path.join(annotations_dir, 'train.txt')
        val_path = os.path.join(annotations_dir, 'val.txt')

        if os.path.exists(train_path) and os.path.exists(val_path):
            return
        print(f"'train.txt' and 'val.txt' not found. Creating a new split with ratio {split_ratio}.")
        if not os.path.exists(trainval_path):
            raise FileNotFoundError(f"'trainval.txt' not found in {annotations_dir}. Cannot create split.")

        with open(trainval_path, 'r') as f:
            lines = f.readlines()

        random.shuffle(lines)

        split_index = int(len(lines) * split_ratio)
        train_lines = lines[:split_index]
        val_lines = lines[split_index:]

        with open(train_path, 'w') as f:
            f.writelines(train_lines)
        print(f"-> Created 'train.txt' with {len(train_lines)} samples.")

        with open(val_path, 'w') as f:
            f.writelines(val_lines)
        print(f"-> Created 'val.txt' with {len(val_lines)} samples.")


    @property
    def mask_to_label(self) -> dict[str, int]:
        if self._mask_to_label is None:
            self._mask_to_label = {'background': 0, 'pet': 1, 'boundary': 2}
        return self._mask_to_label

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        trimap = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = np.zeros(trimap.shape, dtype=np.uint8)
        mask[trimap == 1] = self.mask_to_label['pet']
        mask[trimap == 3] = self.mask_to_label['boundary']

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


if __name__ == '__main__':
    random.seed(42)
    dataset_root = DATASETS_PATH / 'pets'
    dataset = OxfordPetBoundaryDataset(root_dir=dataset_root, transform=get_transforms(256, 'val'), split='val')
    ds_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for image_tensor, mask_tensor in ds_loader:
        masks = visualize_mask(image_tensor, mask_tensor)
        image_to_show = masks.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(image_to_show)
        plt.axis('off')
        plt.show()


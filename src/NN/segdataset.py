from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torch
from torchvision import transforms


class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 target_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 fraction_train: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 target_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            target_folder (str): Name of the folder that contains the target in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and validation split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Val' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            target_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.

        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Val'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        target_folder_path = Path(self.root) / target_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not target_folder_path.exists():
            raise OSError(f"{target_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if target_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{target_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.target_color_mode = target_color_mode

        if fraction and not fraction_train:
            if subset not in ["Train", "Val"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Val."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.target_list = np.array(sorted(target_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.target_list = self.target_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[:int(np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.target_names = self.target_list[:int(np.ceil(len(self.target_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.target_names = self.target_list[int(np.ceil(len(self.target_list) * (1 - self.fraction))):]
        elif not fraction and fraction_train:
            if subset not in ["Train", "Val"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Val."
                ))
            self.fraction_train = fraction_train
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.target_list = np.array(sorted(target_folder_path.glob("*")))
            if subset == "Train":
                self.image_names = self.image_list[:int(np.ceil(len(self.image_list) * self.fraction_train))]
                self.target_names = self.target_list[:int(np.ceil(len(self.target_list) * self.fraction_train))]
            else:
                self.image_names = sorted(image_folder_path.glob("*"))
                self.target_names = sorted(target_folder_path.glob("*"))


    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        target_path = self.target_names[index]
        with open(image_path, "rb") as image_file, open(target_path, "rb") as target_file:
            image = Image.open(image_file)
            image = image.resize((224, 224))
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")

            target = Image.open(target_file)
            target = target.resize((224, 224))
            if self.target_color_mode == "rgb":
                target = target.convert("RGB")
            elif self.target_color_mode == "grayscale":
                target = target.convert("L")

            transform = transforms.Compose([transforms.Resize(224), transforms.PILToTensor()])
            target = transform(target)

            sample = {"image": image, "target": target}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                # sample["target"] = self.transforms(sample["target"])
            return sample

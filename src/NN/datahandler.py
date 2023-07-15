from pathlib import Path
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from segdataset import SegmentationDataset


def get_dataloader_sep_folder(data_dir: str,
                              image_folder: str = 'Images',
                              target_folder: str = 'Target',
                              fraction_train: int = 1,
                              batch_size: int = 4):
    """ Create Train and Val dataloaders from two
        separate Train and Val folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Images
        ---------Image1
        ---------ImageN
        ------Target
        ---------Target1
        ---------TargetN
        --Val
        ------Images
        ---------Image1
        ---------ImageM
        ------Target
        ---------Target1
        ---------TargetM

    Args:
        data_dir (str): The data directory or root.
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        target_folder (str, optional): Target folder name. Defaults to 'Target'.
        batch_size (int, optional): Batch size of the dataloader. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Val dataloaders.
    """

    data_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir) / x,
                               transforms=data_transforms,
                               image_folder=image_folder,
                               fraction_train=fraction_train,
                               subset=x,
                               target_folder=target_folder)
        for x in ['Train', 'Val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True,
                      num_workers=8)
        for x in ['Train', 'Val']
    }
    return dataloaders


def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str = 'Images',
                                 target_folder: str = 'Target',
                                 fraction: float = 0.4,
                                 batch_size: int = 4):
    """Create train and validation dataloader from a single directory containing
    the image and mask folders.

    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        target_folder (str, optional): Target folder name. Default to 'Target'.
        fraction (float, optional): Fraction of Val set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Val dataloaders.
    """
    data_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               target_folder=target_folder,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True,
                      num_workers=8)
        for x in ['Train', 'Val']
    }
    return dataloaders

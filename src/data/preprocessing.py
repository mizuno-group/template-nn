# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

pytorchでのデータセットの実装のテンプレート.
基本的にはMyDatasetクラスを実装し, DataLoaderを作成する関数があればよい.

@author: tadahaya
"""
import numpy as np
from typing import Tuple, Optional, List

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# 必須
class MyDataset(Dataset):
    """
    Custom dataset implementation for supervised and unsupervised tasks.

    Parameters
    ----------
    data : np.ndarray
        Array containing the data samples.

    label : Optional[np.ndarray]
        Array containing labels for supervised learning.
        Defaults to `None` for unsupervised learning.

    transform : Optional[callable]
        Transformation function to apply to the data samples.
    """
    def __init__(
        self,
        data:np.ndarray=None,
        label:Optional[np.ndarray]=None,
        transform:Optional[callable]=None
        ) -> None:
        if data is None:
            raise ValueError("`data` cannot be None. Please provide the input data.")
        if label is None:
            label = np.full(len(data), np.nan)  # Assign NaN for unsupervised learning
        if not isinstance(transform, list):
            self.transform = [transform]
        else:
            self.transform = transform
        self.data = data
        self.label = label
        self.datanum = len(self.data)

    def __len__(self) -> int:
        """ Returns the number of samples in the dataset. """
        return self.datanum

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Retrieves a single data sample and its corresponding label.
        
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            Tuple[torch.Tensor, float]: Transformed data sample and its label.
        """
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            for t in self.transform:
                if t is not None:
                    out_data = t(out_data)
        return out_data, out_label


def prep_dataloader(
    dataset:Dataset=None,
    batch_size:int=None,
    shuffle:Optional[bool]=None,
    num_workers:int=2,
    pin_memory:bool=True,
    g:Optional[torch.Generator]=None,
    seed_worker:Optional[callable]=None
    ) -> DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=seed_worker,
        )    
    return loader


# オプション
class SubsetWrapper(Dataset):
    """
    Wrapper class for creating a subset of a given dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Original dataset from which to create the subset.

    transform : Optional[callable]
        Transformation function to apply to the data samples.

    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def split_dataset(
    full_dataset:Dataset, split_ratio:float=0.8, shuffle:bool=True,
    transform:Tuple[Optional[List[callable]], Optional[List[callable]]]=(None, None),
) -> Tuple[Dataset, Dataset]:
    """
    Splits a dataset into training and validation sets.

    Parameters
    ----------
    full_dataset : torch.utils.data.Dataset
        The dataset to split.

    split_ratio : float
        The ratio of the dataset to use for training.

    shuffle : bool
        Whether to shuffle the data before splitting.

    transform : Tuple[Optional[List[callable]], Optional[List[callable]]]
        Transformations to apply to the training and validation datasets.
        The first element is for the training dataset and the second is for the validation dataset.
 
    """
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    # transformの適用
    if transform[0]:
        train_dataset = SubsetWrapper(train_dataset, transform[0])
    if transform[1]:
        val_dataset = SubsetWrapper(val_dataset, transform[1])
    return train_dataset, val_dataset
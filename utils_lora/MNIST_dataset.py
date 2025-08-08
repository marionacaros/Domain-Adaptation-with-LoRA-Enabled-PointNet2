import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np

from pytorch_lightning import Callback
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split, Dataset

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

from torchmetrics import Accuracy

import pandas as pd
import seaborn as sn
import math
import matplotlib.pyplot as plt

# set precision to what lightning suggests for this gpu
torch.set_float32_matmul_precision('high')


def transform_img2pc(img, number_of_points=200):
    img_array = np.asarray(img)  # (28,28)
    
    indices = np.argwhere(img_array > 127)
    pc = indices.astype(np.float32)
    if number_of_points - pc.shape[0] > 0:
        # Duplicate points
        sampling_indices = np.random.choice(pc.shape[0], number_of_points - pc.shape[0])
        new_points = pc[sampling_indices, :]
        pc = np.concatenate((pc, new_points), axis=0)
    else:
        sampling_indices = np.random.choice(pc.shape[0], number_of_points)
        pc = pc[sampling_indices, :]
    
    # Add z dimension with noise
    noise = np.random.normal(0, 0.05, len(pc))
    noise = np.expand_dims(noise, 1)
    pc = np.hstack([pc, noise]).astype(np.float32)

    # Normalize the point cloud to have values between 0 and 1 along each axis
    min_vals = pc.min(axis=0)
    max_vals = pc.max(axis=0)
    pc = (pc - min_vals) / (max_vals - min_vals)
    
    return torch.tensor(pc)


class MNISTPointCloudDataset(Dataset):
    def __init__(self, mnist_dataset, transform=None, number_of_points=200):
        self.mnist_dataset = mnist_dataset
        self.transform = transform
        self.number_of_points = number_of_points

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, target = self.mnist_dataset[idx]
        pc = transform_img2pc(img, self.number_of_points)
        if self.transform:
            pc = self.transform(pc)
        return pc, target
    @property
    def targets(self):
        return self.mnist_dataset.targets


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=1024, number_of_points=300, class_names= [0, 1, 2, 3, 4, 5, 6, 7],
                target_labels_to_modify=[7, 8, 9], new_target_label=7):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.number_of_points = number_of_points
        self.class_names =class_names
        self.min_class = min(self.class_names)
        self.num_classes = len(self.class_names)
        self.target_labels_to_modify = target_labels_to_modify
        self.new_target_label = new_target_label

    def prepare_data(self):
        # download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=None)
            mnist_full.targets = mnist_full.targets.clone().detach()
            
            # Modify the targets based on the provided parameters
            for label in self.target_labels_to_modify:
                mnist_full.targets[mnist_full.targets == label] = self.new_target_label

            mask = (mnist_full.targets >= self.min_class) & (mnist_full.targets <= max(self.class_names))
            mnist_full.data = mnist_full.data[mask]
            mnist_full.targets = mnist_full.targets[mask]
            
            # Calculate new split lengths
            num_samples = len(mnist_full)
            train_length = int(num_samples * 0.6)
            val_length = num_samples - train_length
            
            self.mnist_train, self.mnist_val = random_split(mnist_full, [train_length, val_length])
            print(f'Train samples: {train_length}, Test samples: {val_length}')
            
            self.mnist_train = MNISTPointCloudDataset(self.mnist_train, number_of_points=self.number_of_points)
            self.mnist_val = MNISTPointCloudDataset(self.mnist_val, number_of_points=self.number_of_points)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=None)
            self.mnist_test.targets = self.mnist_test.targets.clone().detach()
            mask = (self.mnist_test.targets >= self.min_class) & (self.mnist_test.targets <= max(self.class_names))
            self.mnist_test.data = self.mnist_test.data[mask]
            self.mnist_test.targets = self.mnist_test.targets[mask]
            self.mnist_test = MNISTPointCloudDataset(self.mnist_test, number_of_points=self.number_of_points)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=16)
    
    
def rotate_point_cloud_z(batch_data, rotation_angle=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction.
        Use input angle if given.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if not rotation_angle:
        rotation_angle = np.random.uniform() * 2 * np.pi

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data

from torch.utils.data import Dataset
import torch
import numpy as np

# Define CIFAR / MNIST dataset
class CifarMNISTDataset(Dataset):
    def __init__(self, data, cifar_targets, digit_targets, background_targets):
        self.data = torch.from_numpy(np.transpose(data, (0, 3, 1, 2)).astype(float)).float()
        self.cifar_targets = torch.from_numpy(cifar_targets.astype(float)).float()
        self.digit_targets = torch.from_numpy(digit_targets.astype(float)).float()
        self.background_targets = torch.from_numpy(background_targets.astype(float)).float()

    def __len__(self):
        return len(self.cifar_targets)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, cifar_target, digit_target, background_target_label = self.data[index], int(self.cifar_targets[index]), int(self.digit_targets[index]), int(self.background_targets[index])

        return img, cifar_target, digit_target, background_target_label

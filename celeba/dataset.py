from torch.utils.data import Dataset
import torch

# Define celeba dataset class
class CelebaDataset(Dataset):
    def __init__(self, data, background_targets):
        self.data = torch.from_numpy(data.astype(float)).float()
        self.background_targets = torch.from_numpy(background_targets.astype(float)).float()

    def __len__(self):
        return len(self.background_targets)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, background_target_label = self.data[index], int(self.background_targets[index])

        return img, background_target_label
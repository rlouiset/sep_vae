from torch.utils.data import Dataset
import torch

# Define pneumonia dataset class
class PneumoniaDataset(Dataset):
    def __init__(self, data, specific_targets, background_targets):
        self.data = torch.from_numpy(data.astype(float)).float()
        self.specific_targets = torch.from_numpy(specific_targets.astype(float)).float()
        self.background_targets = torch.from_numpy(background_targets.astype(float)).float()

    def __len__(self):
        return len(self.specific_targets)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, specific_target, background_target_label = self.data[index], int(self.specific_targets[index]), int(self.background_targets[index])

        return img, specific_target, background_target_label
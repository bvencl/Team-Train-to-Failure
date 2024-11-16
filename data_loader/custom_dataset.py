from abc import abstractmethod
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None, **kwargs):
        assert len(data) == len(labels), "Length of data and labels must be the same!"

        if not isinstance(data, np.ndarray):
            labels = np.array(labels)
                              
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        self.data = [data[i] for i in indices]
        self.labels = labels[indices]

        self.transforms = transform
        if interpretable_labels is not None:
            self.interpretable_labels = interpretable_labels[indices]
        else:
            self.interpretable_labels = self.labels

        if transform is None:
            self.transforms = transforms.ToTensor()
            
        self.data = [self.transforms(d) for d in self.data]
        self.labels = [torch.tensor(l) for l in self.labels]

    def __len__(self):
        return len(self.data)

    @abstractmethod    
    def __getitem__(self, idx):
        """Dataset dependent"""

    @abstractmethod
    def get_original_image(self, idx):
        """Dataset dependent"""


class BirdClefDataset(CustomDataset):
    def __init__(self, data, labels, position, transform=None, interpretable_labels=None, files=None):
        super().__init__(data, labels, transform, interpretable_labels)
        self.positions = torch.tensor(position)
        self.files = files

    def __getitem__(self, idx):
        img_data = self.data[idx]
        label = self.labels[idx]
        position = self.positions[idx]

        if self.files is not None:
            file = self.files[idx]

        return img_data, label, position, file


    def get_original_image(self, idx):
        img_data = self.data[idx]
        return Image.fromarray(img_data)
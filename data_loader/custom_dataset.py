import os

import torch
import numpy as np

class BirdClefDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_df, config, label2idx, idx2label, transform=None):
        """
        Initializes the dataset.

        Args:
            metadata_df: DataFrame containing metadata for the dataset. Made by the AudioPreprocessor.
            config: Configuration object with data-related parameters.
            transform: Callable, to apply transformations to the data. Made by the TransformFactory.
        """
        self.metadata_df = metadata_df
        self.transform = transform
        self.cwd = os.getcwd()

        self.label2idx = label2idx
        self.idx2label = idx2label
        
    def print_all_labels(self):
        '''
        Function for debugging purposes. Prints all the labels and their corresponding indices.
        '''
        for label, idx in zip(self.metadata_df["label"], [self.label2idx[label] for label in self.metadata_df["label"]]):
            print(f"Label: {label}, Index: {idx}")

    def print_labels(self):
        """
        Function for debugging purposes. Prints all the labels and their corresponding indices.
        """
        for label, idx in self.label2idx.items():
            print(f"Label: {label}, Index: {idx}")

    def __len__(self):
        return len(self.metadata_df)

    def load_and_transform(self, file_path):
        """
        Loads and augments the spectrogram from a file.

        Args:
            file_path: Path to the .npy file.

        Returns:
            Transformed spectrogram.
        """
        # Load the spectrogram file
        spectrogram = np.load(os.path.join(self.cwd, file_path))
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        # Apply transform if available
        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram

    def __getitem__(self, idx):
        """
        Fetches the item at the given index.

        Args:
            idx: Index of the item to fetch.

        Returns:
            Transformed data and corresponding label index.
        """
        row = self.metadata_df.iloc[idx]
        file_path = row["segment_path"]
        label = row["label"]

        # Convert label to index using precomputed mapping
        label_idx = self.label2idx[label]

        # Load and transform the spectrogram
        spec = self.load_and_transform(file_path)

        return spec, torch.tensor(label_idx, dtype=torch.int64)

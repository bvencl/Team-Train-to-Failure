import os
import torch
import torchaudio
import numpy as np

class BirdClefDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_df, config, transform=None):
        """
        Initializes the dataset.

        Args:
            metadata_df: DataFrame containing metadata for the dataset. Made by the AudioPreprocesser.
            config: Configuration object with data-related parameters.
            transform: Callable, to apply transformations to the data. Made by the TransformFactory.
        """
        self.metadata_df = metadata_df
        self.transform = transform


    def __len__(self):
        return len(self.metadata_df)

    def load_and_transform(self, file_path):
        """
        Loads and preprocesses the audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transformed waveform.
        """
        # Load the audio file
        spectrogram = np.load(file_path)
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
            Transformed data and corresponding label.
        """
        row = self.metadata_df.iloc[idx]
        file_path = row["segment_path"]
        label = row["label"]

        waveform = self.load_and_transform(file_path)

        label_idx = self.metadata_df["label"].unique().tolist().index(label)

        return waveform, torch.tensor(label_idx, dtype=torch.int64)


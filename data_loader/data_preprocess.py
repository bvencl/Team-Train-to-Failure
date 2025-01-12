import os
import hashlib

import torch
import torchaudio
import torchaudio.transforms as T
import torchvision

import librosa as lb

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_loader.load_metadata import load_metadata, load_testing_data


class AudioPreprocesser:
    """
    Class for preprocessing the audio data.
    """

    def __init__(self, config, visualiser=None):
        self.config = config
        self.sample_rate = config.data_process.sample_rate
        self.desired_length_s = config.data_process.max_length_s
        self.desired_length = int(self.sample_rate * self.desired_length_s)
        self.visualiser = visualiser
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pad_audio = (
            self._pad_end
            if self.config.data_process.pad_mode == "end"
            else self._pad_center
        )
        self.load_data = (
            load_metadata if not self.config.testing.testing else load_testing_data
        )
        self._make_mel_spectrogram = (
            self._make_mel_spectrogram_librosa
            if self.config.data_process.use_librosa
            else self._make_mel_spectrogram_torch
        )

    def process_database(self):
        """
        Processes the audio database: checks if the data is already processed, if not than processes it.
        Calls the _process_audio method for each row in the metadata.
        Can be parallelized for faster data processing.
        """

        # making a hash of the configuration to check if there was a change in the configuration, so the data needs to be reprocessed
        config_hash = self._compute_config_hash()
        hash_file_path = os.path.join(self.config.data.output_dir, "config_hash.txt")
        final_metadata_path = os.path.join(
            self.config.data.output_dir, "final_metadata.csv"
        )

        metadata = self._load_data()
        if metadata.empty:
            raise ValueError("The metadata DataFrame is empty. Check your input data.")

        os.makedirs(self.config.data.output_dir, exist_ok=True)

        # Check if metadata is already processed
        if os.path.exists(final_metadata_path) and os.path.exists(hash_file_path):
            with open(hash_file_path, "r") as f:
                saved_hash = f.read().strip()

            # If the hash is the same, the metadata is already processed with the same configuration
            if saved_hash == config_hash:

                print("Metadata already processed. Loading existing metadata...")
                full_meta_df = pd.read_csv(final_metadata_path)
                if self.config.testing.testing:
                    full_meta_df = full_meta_df[
                        : self.config.testing.data_samples_for_testing
                    ]

                # Keep only the rows with the primary_label
                full_meta_df = full_meta_df[
                    full_meta_df.groupby("label")["label"].transform("count")
                    >= self.config.data.min_samples_in_class
                ]

                # Create label2idx and idx2label dictionaries for encoding the labels
                label2idx = {
                    label: idx
                    for idx, label in enumerate(full_meta_df["label"].unique())
                }
                idx2label = {idx: label for label, idx in label2idx.items()}

                # Split the data into train, validation and test sets
                train_df, val_df, test_df = self._split_data(full_meta_df)

                return train_df, val_df, test_df, label2idx, idx2label

        # If the hash is different, the metadata needs to be reprocessed (Either the configuration changed or there was no existing metadata/processed data)
        else:
            # save the hash of the current configuration
            with open(hash_file_path, "w") as f:
                f.write(config_hash)

        print("Processing metadata from scratch...")

        # Subset for testing if enabled
        if self.config.testing.testing:
            metadata = metadata[: self.config.testing.data_samples_for_testing]
            print(len(metadata))

        all_metadata = []

        # Enable parallel processing
        if self.config.data.multi_threading:
            print("Using parallel processing...")
            from joblib import Parallel, delayed

            # Explicitly create a list of indices to process
            metadata_indices = range(len(metadata))

            def process_row_wrapper(idx):
                if idx < 0 or idx >= len(metadata):
                    return None
                return self._process_audio(idx, metadata)

            # Process in parallel
            results = Parallel(n_jobs=self.config.data.num_workers)(
                delayed(process_row_wrapper)(idx)
                for idx in tqdm(metadata_indices, total=len(metadata_indices))
            )
            all_metadata.extend(results)

        # Single-threaded processing
        else:
            print("Using single-threaded processing...")
            for idx in tqdm(metadata.index, desc="Processing audio files"):
                if idx < 0 or idx >= len(metadata):
                    print(
                        f"Error processing the {idx}th file! Continuing without it..."
                    )
                    continue
                all_metadata.append(self._process_audio(idx, metadata))

        # Merge and save metadata
        full_meta_df = pd.concat(all_metadata, ignore_index=True)
        full_meta_df.to_csv(final_metadata_path, index=False)

        # Keep only the rows with the primary_label
        full_meta_df = full_meta_df[
            full_meta_df.groupby("label")["label"].transform("count")
            >= self.config.data.min_samples_in_class
        ]

        # Create label2idx and idx2label dictionaries for encoding the labels
        label2idx = {
            label: idx for idx, label in enumerate(full_meta_df["label"].unique())
        }
        
        idx2label = {idx: label for label, idx in label2idx.items()}

        print(f"Metadata saved to {final_metadata_path}.")

        # Split the data into train, validation and test sets
        train_df, val_df, test_df = self._split_data(full_meta_df)
        
        return train_df, val_df, test_df, label2idx, idx2label

    def _split_data(self, full_meta_df):
        """
        Split data for train, validation and test
        """

        train_df, val_test_df = train_test_split(
            full_meta_df,
            test_size=1 - self.config.data.train_ratio,
            stratify=full_meta_df["label"],
            random_state=self.config.trainer.seed,
        )

        if self.config.data.test_val_ratio == 0:
            val_df = val_test_df
            test_df = None
            return train_df, val_df, test_df
        
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=self.config.data.test_val_ratio,
            stratify=val_test_df["label"],
            random_state=self.config.trainer.seed,
        )

        return train_df, val_df, test_df

    def _load_data(self):
        """
        Returns the existing (unprocessed) metadata
        """
        return self.load_data(self.config)

    def _process_audio_wrapper(self, row):
        """
        A helper function that extracts the elements of an iterable object and calls _process_audio
        """
        idx, data = row
        return self._process_audio(idx, data)

    def _process_audio(self, idx, dataframe):
        """
        Create the right identical structure for sound files, like sample rate, mono, and size.

        Creating melspectrograms and saving them to disk.
        """
        idx_row = dataframe.iloc[idx]

        path = idx_row["file_path"]

        y, sample_rate = torchaudio.load(path)

        y = self._unifiy_sample_rate(sample_rate, y)

        y = self._make_mono(y)

        meta_data_list = []
        # Generate sound files of the same size and the melspectrograms that can be created from them
        for i, y in enumerate(self._resize_audio(y)):
            mel_spec = self._make_mel_spectrogram(y)

            if self.config.data_process.resize_method == "function_resize":
                resize_transform = torchvision.transforms.Resize((224, 224))
                mel_spec = resize_transform(mel_spec)

            mel_spec = mel_spec.expand(3, -1, -1)

            # !mel_spec = self._normalise_standardise(mel_spec) # Use here is depracted!

            segment_path = self._save_segment(
                mel_spec, path, idx_row, i
            )

            meta_data_list.append(
                {
                    "file_id": os.path.splitext(os.path.basename(path))[0],
                    "segment_index": i,
                    "label": idx_row.get("primary_label", None),
                    "latitude": idx_row.get("latitude", None),
                    "longitude": idx_row.get("longitude", None),
                    "original_path": path,
                    "segment_path": segment_path,
                    "segment_length": y.shape[1] / self.sample_rate,
                    "spectrogram_length": mel_spec.shape[1]
                    * self.config.data_process.hop_length
                    / self.sample_rate,
                }
            )

        _file_df = pd.DataFrame(meta_data_list)

        return _file_df

    def _unifiy_sample_rate(self, sr, y):
        """
        Resample the audio if it's not at the desired sample rate.
        """
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            y = resampler(y)
        return y

    def _make_mono(self, y):
        """
        Converts a multi-channel (e.g. stereo) sound sample into a mono sound sample
        """
        if y.shape[0] > 1:
            y = torch.mean(y, dim=0, keepdim=True)
        return y

    def _resize_audio(self, y):
        """
        Resize the audio based on the mode in the configuration.

        ## Modes:
        - 'slice': Slice the audio into segments of `desired_length`.
        - 'single': Keep only the beginning of the audio or pad if necessary.

        ## Args:
            y (Tensor): The input audio waveform.

        ## Yields:
            Tensor: Resized audio segments.
        """
        num_samples = y.shape[1]

        if self.config.data_process.mode == "slice":
            start = 0
            while start + self.desired_length <= num_samples:
                yield y[:, start : start + self.desired_length]
                start += self.desired_length

            if start < num_samples:
                # Handle the remaining segment (pad if necessary)
                remaining_segment = y[:, start:]
                if remaining_segment.shape[1] < self.desired_length:
                    remaining_segment = self._pad_audio(remaining_segment)
                yield remaining_segment

        elif self.config.data_process.mode == "single":
            # Keep only the first segment, pad if necessary
            if num_samples < self.desired_length:
                y = self._pad_audio(y)
            yield y[:, : self.desired_length]

        else:
            raise ValueError(
                f"Invalid mode: {self.config.data_process.mode}. Not slice or single."
            )

    def _pad_end(self, y):
        """
        If the audio sample is shorter than desired, it will be extended by padding
        """
        num_samples = y.shape[1]
        pad_length = self.desired_length - num_samples

        if self.config.data_process.pad_values == "zeros":
            return torch.nn.functional.pad(y, (0, pad_length), mode="constant", value=0)

        elif self.config.data_process.pad_values == "repeat":

            repeat_pad = y[:, -1:].repeat(1, pad_length)
            return torch.cat((y, repeat_pad), dim=1)

        else:
            raise ValueError(
                f"Invalid pad_values: {self.config.data_process.pad_values}. Not zeros or repeat."
            )

    def _pad_center(self, y):
        """
        If the audio sample is shorter than desired, it will be extended by padding
        """
        num_samples = y.shape[1]
        pad_length = self.desired_length - num_samples
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        if self.config.data_process.pad_values == "zeros":
            return torch.nn.functional.pad(
                y, (left_pad, right_pad), mode="constant", value=0
            )

        elif self.config.data_process.pad_values == "repeat":
            print("Repeat padding")
            left_repeat = y[:, :1].repeat(1, left_pad)
            right_repeat = y[:, -1:].repeat(1, right_pad)
            return torch.cat((left_repeat, y, right_repeat), dim=1)

        else:
            raise ValueError(
                f"Invalid pad_values: {self.config.data_process.pad_values}. Not 'zeros' or 'repeat'."
            )

    def _make_mel_spectrogram_torch(self, y):
        """
        Making the mel spectrogram from the audio
        """
        y = y.to(self.device)
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.data_process.n_mels,
            n_fft=self.config.data_process.n_fft,
            hop_length=self.config.data_process.hop_length,
            f_min=self.config.data_process.f_min,
            f_max=self.config.data_process.f_max,
        ).to(self.device)
        to_db = T.AmplitudeToDB().to(self.device)

        mel_spec = mel_transform(y)
        mel_spec = to_db(mel_spec)

        mel_spec = self._normalise_standardise(mel_spec)

        return mel_spec.to("cpu")

    def _make_mel_spectrogram_librosa(self, y):
        """
        Making the mel spectrogram from the audio
        """
        y = y.numpy()
        spec = lb.feature.melspectrogram(
            y=y,
            sr=self.config.data_process.sample_rate,
            n_mels=self.config.data_process.n_mels,
            n_fft=self.config.data_process.n_fft,
            hop_length=self.config.data_process.hop_length,
            fmax=self.config.data_process.f_max,
            fmin=self.config.data_process.f_min,
        )
        spec = lb.power_to_db(spec, ref=1.0)
        min_ = spec.min()
        max_ = spec.max()
        if max_ != min_:
            spec = (spec - min_) / (max_ - min_)

        return torch.tensor(spec).to("cpu")

    def _normalise_standardise(self, y):
        """
        Standardise and normalise the mel spectrograms if enabled in the configuration, can be called either way.
        """
        if self.config.data_process.normalise:
            y = self._normalise(y)

        if self.config.data_process.standardise:
            y = self._standardise(y)
            
        return y

    def _standardise(self, y):
        """
        Standardise the mel spectrograms
        """
        if (
            self.config.model.type == "efficientnet_v2_m"
            or self.config.model.type == "efficientnet_v2_s"
            or self.config.model.type == "mobilenet_v3_small"
            or self.config.model.type == "mobilenet_v3_large"
        ):
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            y = (y - mean) / std
            return y.to("cpu")
        
        else:
            y = y.to(self.device)
            mean = y.mean()
            std = y.std()
            y = (y - mean) / (std + 1e-8)
            return y.to("cpu")

    def _normalise(self, y):
        """
        Normalise the mel spectrograms
        """
        y_min = y.min()
        y_max = y.max()
        y = (y - y_min) / (y_max - y_min + 1e-8)

        return y.to("cpu")

    def _save_segment(self, y, path, df_row, segment_index):
        """
        Saves the segmented spectrograms in .npy files as numpy arrays
        """
        base_dir = self.config.data.output_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        file_id = os.path.splitext(os.path.basename(path))[0]
        save_dir = os.path.join(base_dir, file_id)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"segment_{segment_index:05d}.npy")

        np.save(save_path, y.numpy())

        return save_path

    def _compute_config_hash(self):
        """
        Compute the hash of the configuration object
        """
        config_str = str(self.config.data_process)
        return hashlib.sha256(config_str.encode()).hexdigest()

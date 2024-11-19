import os
import hashlib

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_loader.load_metadata import load_metadata

from concurrent.futures import ProcessPoolExecutor


class AudioPreprocesser:
    def __init__(self, config, visualiser=None):
        self.config = config
        self.sample_rate = config.data_process.sample_rate
        self.desired_length_s = config.data_process.max_length_s
        self.desired_length = int(self.sample_rate * self.desired_length_s)
        self.visualiser = visualiser
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_database(self):
        config_hash = self._compute_config_hash()
        hash_file_path = os.path.join(self.config.data.output_dir, "config_hash.txt")
        final_metadata_path = os.path.join(
            self.config.data.output_dir, "final_metadata.csv"
        )

        os.makedirs(self.config.data.output_dir, exist_ok=True)

        if os.path.exists(final_metadata_path) and os.path.exists(hash_file_path):
            with open(hash_file_path, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == config_hash:
                print("Metadata already processed. Loading existing metadata...")
                
                full_meta_df = pd.read_csv(final_metadata_path)
                
                return self._split_data(full_meta_df)
        else:
            with open(hash_file_path, "w") as f:
                f.write(config_hash)

        print("Processing metadata from scratch...")
        metadata = self._load_data()
        if metadata.empty:
            raise ValueError("The metadata DataFrame is empty. Check your input data.")

        if self.config.testing.testing:
            metadata = metadata[:self.config.testing.data_samples_for_testing]

        all_metadata = []

        if self.config.data.multi_threading:
            with ProcessPoolExecutor(max_workers=self.config.data.num_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(self._process_audio_wrapper, metadata.iterrows()),
                        total=len(metadata),
                        desc="Processing audio files",
                    )
                )
                all_metadata.extend(results)
        else:
            for idx, row in tqdm(metadata.iterrows(), desc="Processing audio files"):
                if idx < 0 or idx >= len(metadata):
                    print(f"Invalid idx: {idx}")
                    continue
                all_metadata.append(self._process_audio(idx, metadata))
                


        full_meta_df = pd.concat(all_metadata, ignore_index=True)

        full_meta_df.to_csv(final_metadata_path, index=False)
        with open(hash_file_path, "w") as f:
            f.write(config_hash)
        print(f"Metadata saved to {final_metadata_path}.")
        
        return self._split_data(full_meta_df)

    def _split_data(self, full_meta_df):
        train_df, val_test_df = train_test_split(
            full_meta_df,
            test_size=1 - self.config.data.train_ratio,
            stratify=full_meta_df["label"],
            random_state=self.config.trainer.seed,
        )

        val_df, test_df = train_test_split(
            val_test_df,
            test_size=self.config.data.test_val_ratio,
            stratify=val_test_df["label"],
            random_state=self.config.trainer.seed,
        )

        return train_df, val_df, test_df

    def _load_data(self):
        return load_metadata(self.config)

    def _process_audio_wrapper(self, row):
        idx, data = row
        return self._process_audio(idx, data)

    def _process_audio(self, idx, dataframe):
        idx_row = dataframe.iloc[idx]

        path = idx_row["file_path"]

        y, sample_rate = torchaudio.load(path)

        y = self._unifiy_sample_rate(sample_rate, y)

        y = self._make_mono(y)

        meta_data_list = []
        for i, y in enumerate(self._resize_audio(y)):
            mel_spec = self._make_mel_spectrogram(y)
            mel_spec = self._standardise_normalise(mel_spec)
            segment_path, segment_length = self._save_segment(
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
                    "spectrogram_length": mel_spec.shape[1] * self.config.data_process.hop_length / self.sample_rate,
                }
            )

        _file_df = pd.DataFrame(meta_data_list)

        return _file_df

    def _unifiy_sample_rate(self, sr, y):
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            y = resampler(y)
        return y

    def _make_mono(self, y):
        if y.shape[0] > 1:
            y = torch.mean(y, dim=0, keepdim=True)
        return y

    def _resize_audio(self, y):
        """
        Resize the audio based on the mode in the configuration.

        Modes:
        - 'slice': Slice the audio into segments of `desired_length`.
        - 'single': Keep only the beginning of the audio or pad if necessary.

        Args:
            y (Tensor): The input audio waveform.

        Yields:
            Tensor: Resized audio segments.
        """
        num_samples = y.shape[1]

        if self.config.data_process.mode == 'slice':
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

        elif self.config.data_process.mode == 'single':
            # Keep only the first segment, pad if necessary
            if num_samples < self.desired_length:
                y = self._pad_audio(y)
            yield y[:, :self.desired_length]
            
        else:
            raise ValueError(f"Invalid mode: {self.config.data_process.mode}. Not slice or single.")


    def _pad_audio(self, y):
        num_samples = y.shape[1]
        pad_length = self.desired_length - num_samples
        return torch.nn.functional.pad(y, (0, pad_length), mode="constant", value=0)

    def _make_mel_spectrogram(self, y):
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
        return mel_spec.to("cpu")

    def _standardise_normalise(self, y):
        y = y.to(self.device)
        mean = y.mean()
        std = y.std()
        y = (y - mean) / (std + 1e-8)
        
        y_min = y.min()
        y_max = y.max()
        y = (y - y_min) / (y_max - y_min + 1e-8)
        
        return y.to("cpu")

    def _save_segment(self, y, path, df_row, segment_index):
        base_dir = self.config.data.output_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        file_id = os.path.splitext(os.path.basename(path))[0]
        save_dir = os.path.join(base_dir, file_id)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"segment_{segment_index:05d}.npy")

        np.save(save_path, y.numpy())

        if self.config.testing.testing:
            print(f"Saved segment {segment_index} for file {path} at {save_path}")

        return save_path, y.shape[1]

    def _compute_config_hash(self):
        config_str = str(self.config.data_process)
        return hashlib.sha256(config_str.encode()).hexdigest()

import matplotlib.pyplot as plt
import librosa as lb
import librosa.display
import numpy as np
import torch
import pandas as pd
import matplotlib as mpl


class Visualiser:
    def __init__(self, config):
        self.config = config

    def load_audio(self, filename):
        """
        Loads the audio file and calculates the spectogram if needed.
        """
        y, sr = lb.load(filename, sr=self.config.data_process.sample_rate)
        mel_spec = lb.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.config.data_process.n_mels,
            hop_length=self.config.data_process.hop_length,
            n_fft=self.config.data_process.n_fft,
            fmin=self.config.data_process.f_min,
            fmax=self.config.data_process.f_max,
        )
        return lb.amplitude_to_db(mel_spec, ref=np.max), sr

    def load_spectrogram(self, filename):
        """
        Loads the mel spectogram.
        """
        return np.load(filename), self.config.data_process.sample_rate

    def visualise(self, y, sr, title="Mel Spectrogram (dB)"):
        """
        Draws the mel spectrogram.
        """
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        if y.ndim == 3:
            y = y[0]

        duration = y.shape[1] * self.config.data_process.hop_length / sr

        plt.figure(figsize=(12, 6))
        librosa.display.specshow(y, 
            sr = self.config.data_process.sample_rate, 
            hop_length=512,
            n_fft=2048,
            fmin=self.config.data_process.f_min,
            fmax=self.config.data_process.f_max,
            x_axis = 'time', 
            y_axis = 'mel',
            cmap = 'coolwarm',
            )
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()

    def __call__(self, y=None, sr=None, filename=None, df_row=None):
        """
        Entry point for the visualiser callback.
        """
        if df_row is not None:
            filename = df_row["segment_path"]

        if filename is not None:
            if "processed_data" in filename:
                y, sr = self.load_spectrogram(filename)
            else:
                y, sr = self.load_audio(filename)

        if y is not None:
            y = y[0]
            if sr is None:
                sr = self.config.data_process.sample_rate
            self.visualise(y, sr, title=f"Mel Spectrogram - {filename if filename else 'Input Data'}")
        else:
            print("What you doing dogg?.")

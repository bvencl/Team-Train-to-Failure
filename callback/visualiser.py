import matplotlib.pyplot as plt
import librosa as lb
import numpy as np
import torch


class Visualiser:
    def __init__(self, config):
        self.config = config

    def __call__(self, y=None, sr=32000, filename=None, df_row=None):
        """
        Displays the mel spectrogram of the given audio or spectrogram data. You can provide either the audio data, or the appropriate row from the pandas DataFrame containing the metadata.

        Parameters:
        y (numpy.ndarray or torch.Tensor, optional): Mel spectrogram data.
        sr (int, optional): Sampling rate of the audio.
        filename (str, optional): Path to the file containing audio or spectrogram.
        df_row (pandas.Series, optional): Row from the metadata DataFrame.
        """

        if df_row is not None:
            filename = df_row["segment_path"]

        if filename is not None and "processed_data" in filename:
            # Load preprocessed spectrogram
            y = np.load(filename)
            sr = self.config.data_process.sample_rate

        elif y is None and filename is not None:
            # Load audio file and compute mel spectrogram
            y, sr = lb.load(filename)
            y = lb.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, hop_length=512, n_fft=2048
            )
            y = lb.amplitude_to_db(y, ref=np.max)

        if y is not None:
            # Convert Torch tensor to NumPy array
            if isinstance(y, torch.Tensor):
                y = y.numpy()

            # Remove the first dimension if the spectrogram has 3 dimensions
            if y.ndim == 3:
                y = y[0]  # Assuming the first dimension is the channel

            # Calculate the duration (only for time-axis visualization)
            duration = (
                y.shape[1]
                * self.config.data_process.hop_length
                / self.config.data_process.sample_rate
            )

            # Plot the spectrogram
            plt.figure(figsize=(duration, 4))
            lb.display.specshow(
                y,
                x_axis="time",
                y_axis="mel",
                sr=sr,
                hop_length=self.config.data_process.hop_length,
                n_fft=self.config.data_process.n_fft,
                fmin=self.config.data_process.f_min,
                fmax=self.config.data_process.f_max,
                cmap="coolwarm",
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(
                f"Mel Spectrogram (dB) - {filename if filename else 'Input Data'}"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            if duration:
                plt.xlim(0, duration)
            plt.tight_layout()
            plt.text(
                0.01,
                0.02,
                f"Duration: {duration:.2f} seconds" if duration else "Unknown duration",
                ha="left",
                va="bottom",
                transform=plt.gca().transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5),
            )
            plt.show()
        else:
            print("The selected spectrogram or audio data is None.")

from utils.data_loader import process_audio
import matplotlib.pyplot as plt
import librosa as lb

# Select a mel spectrogram to display
  # Index for the specific spectrogram

# Check if the selected mel spectrogram exists



class Visualizer():
    def __init__(self):
        ...
        
    def show_sound(self, random_row):
        mel_spectogram = random_row['mel_spectrogram']
        if mel_spectogram is not None:
            plt.figure(figsize=(10, 4))
            lb.display.specshow(mel_spectogram, sr=32000, x_axis="time", y_axis="mel", fmax=16000)
            plt.colorbar(format="%+2.0f dB")
            plt.title("Mel Spectrogram (dB)")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("The selected spectrogram is None.")

    def play_sound(self, random_row):
        pass

    def show_and_play(self, data_frame):
        random_row = data_frame.sample(n=1).iloc[0]
        self.play_sound(random_row)
        self.show_sound(random_row)
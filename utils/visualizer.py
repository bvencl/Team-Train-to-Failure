import matplotlib.pyplot as plt
import librosa as lb
import numpy as np
import subprocess


class Visualizer():
    def __init__(self):
        pass
        
        
    def show_sound(self, random_row):
        mel_spectogram = random_row['mel_spectrogram']
        filename = random_row['filename']
        filepath = f"data/train_audio/{filename}"
        audio, sr = lb.load(filepath, sr=None)
        duration = len(audio) / sr
        
        mel_spectogram = lb.feature.melspectrogram(y=audio, sr=sr)
        mel_spectogram_db = lb.power_to_db(mel_spectogram, ref=np.max)

        if mel_spectogram is not None:
            plt.ion()
            plt.figure(figsize=(10, 4))
            lb.display.specshow(mel_spectogram_db, sr=sr, x_axis="time", y_axis="mel", fmax=sr / 2)
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Mel Spectrogram (dB) - {filename}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.xlim([0, duration])
            plt.tight_layout()
            plt.show()
            plt.draw()
            plt.pause(0.001)
        else:
            print("The selected spectrogram is None.")

    def play_sound(self, random_row):
        filename = random_row['filename']
        filepath = f"data/train_audio/{filename}"
        subprocess.run(['ffplay', '-nodisp', '-autoexit', filepath])
        plt.ioff()
        plt.show()

    def show_and_play(self, data_frame):
        random_row = data_frame.sample(n=1).iloc[0]
        self.show_sound(random_row)
        self.play_sound(random_row)
        print(random_row)

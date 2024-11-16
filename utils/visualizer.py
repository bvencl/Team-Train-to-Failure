import matplotlib.pyplot as plt
import librosa as lb
import subprocess
import threading

class Visualizer():
    def __init__(self, data_frame):
        self.random_row = data_frame.sample(n=1).iloc[0]
        self.filename =  self.random_row['filename']
        self.filepath = f"data/train_audio/{self.filename}"
        self.mel_spectogram = self.random_row['mel_spectogram']
        self.audio, self.sr = lb.load(self.filepath, sr=32000)
        self.duration = len(self.audio) / self.sr
        
        
    def show_sound(self):
        """
        Displays the mel spectrogram of the audio file specified in the random_row.
        
        Parameters:
        random_row (pd.Series): A row from the DataFrame containing the mel spectrogram and filename.
        """

        if self.mel_spectogram is not None: # Check if the spectrogram is not None
            plt.figure(figsize=(10, 4))
            lb.display.specshow(self.mel_spectogram, x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Mel Spectrogram (dB) - {self.filename}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.xlim(0, self.duration)
            plt.tight_layout()
            plt.text(0.01, 0.02, f"Duration: {self.duration:.2f} seconds", ha='left', va='bottom',
                     transform=plt.gca().transAxes, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.5))
            plt.show()
        else:
            print("The selected spectrogram is None.")


    def play_sound(self):
        """
        Plays the audio file specified in the random_row using ffplay.
        
        Parameters:
        random_row (pd.Series): A row from the DataFrame containing the filename.
        """
        subprocess.run(['cvlc', '--play-and-exit', self.filepath])
        # subprocess.run(['ffplay', '-nodisp', '-autoexit', self.filepath])

    def show_and_play(self):
        """
        Selects a random row from the DataFrame, displays the mel spectrogram, and plays the audio.
        
        Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the mel spectrograms and filenames.
        """
        play_thread = threading.Thread(target=self.play_sound)
        play_thread.start()
        self.show_sound()
        play_thread.join()

        plt.ioff()
        plt.show()

        print(self.random_row)
        print(f"Duration: {self.duration} seconds")

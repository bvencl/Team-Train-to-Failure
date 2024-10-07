import os
import pandas as pd
import librosa as lb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

traindata_path = os.getcwd() + "/data/train_audio"
csv_path = os.getcwd() + "/data/train_metadata.csv"

columns_to_read = [
    "primary_label",
    "secondary_labels",
    "type",
    "latitude",
    "longitude",
    "filename",
]

df = pd.read_csv(csv_path, usecols=columns_to_read)

train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df["primary_label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["primary_label"])

def process_audio(df, data_path, scaler=None, fit_scaler=False):
    mel_spectrograms = []

    for filename in df["filename"]:
        audio_path = os.path.join(data_path, filename)
        
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            mel_spectrograms.append(None)
            continue
        
        audio, sr = lb.load(audio_path, sr=32000)
        mel_spec = lb.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr/2)
        
        if fit_scaler:
            mel_spec = scaler.fit_transform(mel_spec)
        else:
            mel_spec = scaler.transform(mel_spec)
        
        mel_spectrograms.append(mel_spec)
    
    return mel_spectrograms

scaler = StandardScaler()
train_mel_spectrograms = process_audio(train_df, traindata_path, scaler=scaler, fit_scaler=True)
val_mel_spectrograms = process_audio(val_df, traindata_path, scaler=scaler, fit_scaler=False)
test_mel_spectrograms = process_audio(test_df, traindata_path, scaler=scaler, fit_scaler=False)

train_df["mel_spectrogram"] = train_mel_spectrograms
val_df["mel_spectrogram"] = val_mel_spectrograms
test_df["mel_spectrogram"] = test_mel_spectrograms

print(train_df.head())
print(val_df.head())
print(test_df.head())
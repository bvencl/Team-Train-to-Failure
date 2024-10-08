import os
import numpy as np
import pandas as pd
import librosa as lb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
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

    primary_label_column = 'primary_label'  # Cseréld le a megfelelő oszlopnévre

    one_hot_encoded_labels = pd.get_dummies(df[primary_label_column], prefix=primary_label_column)

    # Hozzáadjuk a one-hot kódolt oszlopokat az eredeti DataFrame-hez
    df = pd.concat([df, one_hot_encoded_labels], axis=1)

    # Eltávolítjuk az eredeti primary label oszlopot, ha szükséges
    df.drop(columns=[primary_label_column], inplace=True)


    df = df[:50]
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=42, stratify=df["primary_label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["primary_label"]
    )
    scaler = StandardScaler()
    train_mel_spectrograms = process_audio(
        train_df, traindata_path, scaler=scaler, fit_scaler=True
    )
    val_mel_spectrograms = process_audio(
        val_df, traindata_path, scaler=scaler, fit_scaler=False
    )
    test_mel_spectrograms = process_audio(
        test_df, traindata_path, scaler=scaler, fit_scaler=False
    )
    train_df["mel_spectrogram"] = train_mel_spectrograms
    val_df["mel_spectrogram"] = val_mel_spectrograms
    test_df["mel_spectrogram"] = test_mel_spectrograms




    return train_df, val_df, test_df


def process_audio(df, data_path, scaler=None, fit_scaler=False):
    mel_spectrograms = []
    for filename in df["filename"]:
        audio_path = os.path.join(data_path, filename)
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            mel_spectrograms.append(None)
            continue
        audio, sr = lb.load(audio_path, sr=32000)
        mel_spec = lb.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr / 2)
        mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db.T
        if fit_scaler:
            mel_spec_db = scaler.fit_transform(mel_spec_db)
        else:
            mel_spec_db = scaler.transform(mel_spec_db)
        mel_spectrograms.append(mel_spec_db)
    return mel_spectrograms

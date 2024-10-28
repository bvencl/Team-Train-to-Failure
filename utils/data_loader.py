import os
import numpy as np
import pandas as pd
import librosa as lb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(config):
    traindata_path = os.path.join(os.getcwd(), config.paths.labeled)
    csv_path = os.path.join(os.getcwd(), config.paths.metadata)
    columns_to_read = [
        "primary_label",
        "secondary_labels",
        "type",
        "latitude",
        "longitude",
        "common_name",
        "filename",
    ]
    df = pd.read_csv(csv_path, usecols=columns_to_read)

    df = df[:100]

    df = df[df.groupby('primary_label')['primary_label'].transform('count') >= config.data.min_samples_in_class]

    one_hot_encoded_labels = pd.get_dummies(
        df["primary_label"], prefix="primary_label"
    ).astype(int)
    df["one_hot_vector"] = one_hot_encoded_labels.values.tolist()
    
    print(df['primary_label'].value_counts())


    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - config.data.train_ratio),
        random_state=config.trainer.seed,
        stratify=df["primary_label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.data.test_val_ratio,
        random_state=config.trainer.seed,
        stratify=temp_df["primary_label"],
    )

    train_mel_spectrograms = process_audio(train_df, traindata_path)
    val_mel_spectrograms = process_audio(val_df, traindata_path)
    test_mel_spectrograms = process_audio(test_df, traindata_path)

    train_mel_spectrograms = standardize_individually(train_mel_spectrograms)
    val_mel_spectrograms = standardize_individually(val_mel_spectrograms)
    test_mel_spectrograms = standardize_individually(test_mel_spectrograms)

    train_df = train_df.assign(mel_spectrogram=train_mel_spectrograms)
    val_df = val_df.assign(mel_spectrogram=val_mel_spectrograms)
    test_df = test_df.assign(mel_spectrogram=test_mel_spectrograms)

    train_df.to_csv("train_df.csv", index=False)

    return train_df, val_df, test_df


def process_audio(df, data_path):
    mel_spectrograms = []
    for filename in df["filename"]:
        audio_path = os.path.join(data_path, filename)
        if not os.path.exists(audio_path):
            print(f"Warning: File not found - {audio_path}")
            continue
        audio, sr = lb.load(audio_path, sr=32000)
        mel_spec = lb.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr / 2)
        mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)
        mel_spectrograms.append(mel_spec_db.T)

    return mel_spectrograms


def standardize_individually(spectrograms):
    standardized_spectrograms = []
    for spec in spectrograms:
        mean = np.mean(spec)
        std = np.std(spec)
        if std < 1e-6:  # Alacsony szórásnál elkerüljük a túlzott normálást
            standardized_spectrograms.append(spec - mean)
        else:
            standardized_spectrograms.append((spec - mean) / std)
    return standardized_spectrograms

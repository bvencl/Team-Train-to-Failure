import os
import numpy as np
import pandas as pd
import librosa as lb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(config):
    """This function reads the csv storing the metadata and the labeled audio files
        Furthermore, this script makes mel-spectograms from the audio files, standardizes them, and
        separates them into train, validation and test databases.
        This function is probably going to be replaced by some Pytorch-specific methods, 
        custom datasets are on the way.
        ### Args
         - config: the config from the main file is all we need
    """
    traindata_path = os.path.join(os.getcwd(), config.paths.labeled) # Getting the paths for the labeled audio
    csv_path = os.path.join(os.getcwd(), config.paths.metadata) # Getting the path for the csv storing the metadata
    columns_to_read = [
        "primary_label",
        "secondary_labels",
        "type",
        "latitude",
        "longitude",
        "common_name",
        "filename",
    ]
    # Your good'ol Pandas dataframe... Gonna cause some trouble
    df = pd.read_csv(csv_path, usecols=columns_to_read)

    # Reducing the size of the dataset while testing.
    df = df[:config.data.data_num_for_testing] 

    # Considering only those classes where we have "enough" data to even split to train, validate and test datasets 
    df = df[df.groupby("primary_label")["primary_label"].transform("count") >= config.data.min_samples_in_class] 

    # One-Hot encoding the labels 
    one_hot_encoded_labels = pd.get_dummies(df["primary_label"], prefix="primary_label").astype(int)
    df["one_hot_vector"] = one_hot_encoded_labels.values.tolist()

    # Splitting the data into train and temporary datasets
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - config.data.train_ratio),
        random_state=config.trainer.seed,
        stratify=df["primary_label"],
    )
    # Splitting the temporary dataset into test and validation
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.data.test_val_ratio,
        random_state=config.trainer.seed,
        stratify=temp_df["primary_label"],
    )

    # Getting the mel-spectograms for all of the audio files
    train_mel_spectrograms = process_audio(train_df, traindata_path)
    val_mel_spectrograms = process_audio(val_df, traindata_path)
    test_mel_spectrograms = process_audio(test_df, traindata_path)
    
    # Standardizing the spectograms. We chose to standardize the individually, see more about this in the README.md
    # We are thinking about this method, but before changing it we want to see it in training
    train_mel_spectrograms = standardize_individually(train_mel_spectrograms)
    val_mel_spectrograms = standardize_individually(val_mel_spectrograms)
    test_mel_spectrograms = standardize_individually(test_mel_spectrograms)

    # Adding the standardized mel spectograms to the pandas dataframe
    train_df = train_df.assign(mel_spectrogram=train_mel_spectrograms)
    val_df = val_df.assign(mel_spectrogram=val_mel_spectrograms)
    test_df = test_df.assign(mel_spectrogram=test_mel_spectrograms)

    # train_df.to_csv("train_df.csv", index=False) # only for testin purposes
    
    # returning with the dataframes
    return train_df, val_df, test_df


def process_audio(df, dataset_path):
    """This function makes the mel-scpectograms from the audio files"""
    # empty list for 
    mel_spectrograms = []
    # Looping through the audio files of the dataset
    for filename in df["filename"]:
        audio_path = os.path.join(dataset_path, filename)
        if not os.path.exists(audio_path):
            print(f"Warning: File not found - {audio_path}")
            continue
        # Using Librosa to make the spectograms
        audio, sr = lb.load(audio_path, sr=32000)
        # STFT makes the magic real
        mel_spec = lb.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr / 2)
        mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)
        mel_spectrograms.append(mel_spec_db.T)

    return mel_spectrograms


def standardize_individually(spectrograms):
    standardized_spectrograms = []
    for spec in spectrograms:
        mean = np.mean(spec)
        std = np.std(spec)
        if std < 1e-6:
            standardized_spectrograms.append(spec - mean)
        else:
            # Some serious standardization actions
            standardized_spectrograms.append((spec - mean) / std)
    return standardized_spectrograms

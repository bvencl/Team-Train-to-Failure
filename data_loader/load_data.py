import os
import numpy as np
import pandas as pd
import librosa as lb
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_birdclef(config):
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
    # Your good'ol Pandas DataFrame... Just to make life easy
    df = pd.read_csv(csv_path, usecols=columns_to_read)
    
     # Reducing the size of the dataset while testing.
    df = df[:config.data.data_num_for_testing]

    # Considering only those classes where we have "enough" data to even split to train, validate and test datasets
    df = df[df.groupby("primary_label")["primary_label"].transform("count") >= config.data.min_samples_in_class]

    # One-Hot encoding the labels
    one_hot_encoded_labels = pd.get_dummies(df["primary_label"], prefix="primary_label").astype(int)
    df["one_hot_vector"] = one_hot_encoded_labels.values.tolist()
    df["one_hot_vector"] = df["one_hot_vector"].apply(lambda x: np.array(x))
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
    
    # Grpouping the data into arrays
    train_labels = np.array(train_df["one_hot_vector"].values)
    train_files = np.array(train_df["filename"].values)
    train_latitude = train_df["latitude"].values.reshape(-1, 1)
    train_longitude = train_df["longitude"].values.reshape(-1, 1)
    train_common_name = np.array(train_df["common_name"].values)

    val_labels = np.array(val_df["one_hot_vector"].values)
    val_files = np.array(val_df["filename"].values)
    val_latitude = val_df["latitude"].values.reshape(-1, 1)
    val_longitude = val_df["longitude"].values.reshape(-1, 1)
    val_common_name = np.array(val_df["common_name"].values)

    test_labels = np.array(test_df["one_hot_vector"].values)
    test_files = np.array(test_df["filename"].values)
    test_latitude = test_df["latitude"].values.reshape(-1, 1)
    test_longitude = test_df["longitude"].values.reshape(-1, 1)
    test_common_name = np.array(test_df["common_name"].values)


    # Standardize the latitude and longitude parameters
    scaler_latitude = StandardScaler()
    scaler_longitude = StandardScaler()

    train_latitude = scaler_latitude.fit_transform(train_latitude)
    train_longitude = scaler_longitude.fit_transform(train_longitude)

    val_latitude = scaler_latitude.transform(val_latitude)
    val_longitude = scaler_longitude.transform(val_longitude)

    test_latitude = scaler_latitude.transform(test_latitude)
    test_longitude = scaler_longitude.transform(test_longitude)

    # Making them easily accessible
    train_position = np.hstack((train_latitude, train_longitude))
    val_position = np.hstack((val_latitude, val_longitude))
    test_position = np.hstack((test_latitude, test_longitude))

    # Getting the mel-spectograms for all of the audio files
    train_spectograms = process_audio(train_files, traindata_path)
    val_spectograms = process_audio(val_files, traindata_path)
    test_spectograms = process_audio(test_files, traindata_path)


    # Standardizing the spectograms. We chose to standardize the individually, see more about this in the README.md
    # We are thinking about this method, but before changing it we want to see it in training
    train_data = standardize_individually(train_spectograms)
    val_data = standardize_individually(val_spectograms)
    test_data = standardize_individually(test_spectograms)

    # 
    train_transforms = []
    val_transforms = []
    test_transforms = []
    
    # We may put some trainsforms here, we'll decide this when we are going to train the model 

    # we'll surely need the ToTensor() transform
    train_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())


    # Storing the data in a dictionary
    birdclef_data = dict()

    # Making sub-directories
    birdclef_data["train"] = dict()

    birdclef_data["train"]["data"] = train_data
    birdclef_data["train"]["labels"] = train_labels
    birdclef_data["train"]["position"] = train_position
    birdclef_data["train"]["common_name"] = train_common_name
    birdclef_data["train"]["files"] = train_files
    birdclef_data["train"]["transforms"] = transforms.Compose(transforms=train_transforms)

    birdclef_data["val"] = dict()

    birdclef_data["val"]["data"] = val_data
    birdclef_data["val"]["labels"] = val_labels
    birdclef_data["val"]["position"] = val_position
    birdclef_data["val"]["common_name"] = val_common_name
    birdclef_data["val"]["files"] = val_files
    birdclef_data["val"]["transforms"] = transforms.Compose(transforms=val_transforms)

    birdclef_data["test"] = dict()

    birdclef_data["test"]["data"] = test_data
    birdclef_data["test"]["labels"] = test_labels
    birdclef_data["test"]["position"] = test_position
    birdclef_data["test"]["common_name"] = test_common_name
    birdclef_data["test"]["files"] = test_files
    birdclef_data["test"]["transforms"] = transforms.transforms.Compose(transforms=test_transforms)


    # returning with beautiful birdsongs
    return birdclef_data


def process_audio(filenames, dataset_path):
    """This function makes the mel-scpectograms from the audio files"""
    mel_spectrograms = []
    for filename in filenames:
        audio_path = os.path.join(dataset_path, filename)
        if not os.path.exists(audio_path):
            print(f"Warning: File not found - {audio_path}")
            continue
        audio, sr = lb.load(audio_path, sr=32000)
        mel_spec = lb.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr / 2)
        mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)
        mel_spectrograms.append(mel_spec_db)

    return mel_spectrograms


def standardize_individually(spectrograms):
    standardized_spectrograms = []
    for spec in spectrograms:
        mean = np.mean(spec)
        std = np.std(spec)
        
        if (std < 1e-6): # Avoid division by zero
            standardized_spectrograms.append(spec - mean)
        else:
            # Some serious standardization actions
            standardized_spectrograms.append((spec - mean) / std)
    return standardized_spectrograms

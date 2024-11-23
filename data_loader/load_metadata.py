import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_metadata(config):
    """
    Loads and processes the metadata CSV file and splits it into train, validation, and test sets.

    Args:
        config: Configuration object containing paths and data-related parameters.

    Returns:
        A dictionary containing train, validation, and test DataFrames.
    """
    # Load the CSV file
    csv_path = os.path.join(os.getcwd(), config.paths.metadata)
    audio_path = os.path.join(os.getcwd(), config.paths.labeled)
    columns_to_read = [
        "primary_label", "type", "latitude", "longitude", "common_name", "filename"
    ]
    df = pd.read_csv(csv_path, usecols=columns_to_read)

    # Keep only the rows with the type
    df["file_path"] = df["filename"].apply(lambda x: os.path.join(audio_path, x))
    df = df[df["file_path"].apply(os.path.exists)]

    return df
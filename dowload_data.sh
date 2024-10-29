#!/bin/bash

COMPETITION_NAME="birdclef-2024"

DESTINATION_FOLDER="data"

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API key not found. Please place kaggle.json in ~/.kaggle/"
    exit 1
fi

mkdir -p "$DESTINATION_FOLDER"

kaggle competitions download -c "$COMPETITION_NAME" -p "$DESTINATION_FOLDER"

ZIP_FILE="$DESTINATION_FOLDER/$COMPETITION_NAME.zip"
if [ -f "$ZIP_FILE" ]; then
    if command -v unzip &> /dev/null; then
        unzip "$ZIP_FILE" -d "$DESTINATION_FOLDER"
        rm "$ZIP_FILE"
        echo "Data downloaded and extracted successfully."
    else
        echo "Error: 'unzip' command is required but not installed."
        exit 1
    fi
else
    echo "Error: Zip file not found. Please check if the competition name is correct."
    exit 1
fi

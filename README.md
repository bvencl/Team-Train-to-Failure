# Team Train to Failure - [BirdCLEF 2024 @ Kaggle](https://www.kaggle.com/competitions/birdclef-2024)

## Developing Team - Train to Failure

*We are aiming to get a **Proposed Grade***

- **Bódi Vencel (VBW5N9)**  
  Faculty of Electrical Engineering and Informatics,  
  Budapest University of Technology and Economics

- **Mitrenga Márk (OLLNTB)**  
  Faculty of Transportation Engineering and Vehicle Engineering,  
  Budapest University of Technology and Economics

***"Train to Failure"** – not because failure is our goal, but because, well... deep learning often has other plans.*


---

## BirdCLEF Competition

The **Kaggle BirdCLEF 2024** competition challenges participants to develop machine learning models capable of identifying bird species from audio recordings. The dataset focuses on birds from the Western Ghats in India, a globally recognized biodiversity hotspot. Through this competition, participants contribute to conservation efforts by advancing tools to monitor endangered and nocturnal bird species.

---

## Project Overview

### Methodology

We employ a Convolutional Neural Network (CNN) trained on mel-spectrograms generated from audio recordings. These spectrograms are carefully processed and standardized to ensure fair feature extraction. Additionally, we aim to compare the CNN's performance with alternative models, such as Wave2Vec, to explore different approaches for birdsong classification.

---

### Repository Structure

- **`train_classifier.py`**: The main script for training the bird species classification model.
- **`dataloader/data_preprocess.py`**: Handles audio preprocessing, mel-spectrogram generation, and metadata organization into a pandas DataFrame. Processes are optimized for saving intermediate results as `.npy` files.
- **`utils/visualise.py`**: Visualization tools for spectrograms and playback of audio recordings.
- **`utils/utils.py`**: A utility script containing commonly used helper functions.
- **`callbacks/trainer.py`**: Implements the training loop, validation, and testing routines.
- **`factory/*`**: Factory scripts for modular and organized code handling.
- **`config.ini`**: Configuration file containing hyperparameters and other project-specific settings.

---

### Dataset

To download the **BirdCLEF 2024 dataset**, you can use one of the following methods:

1. **Using the `download_data.sh` script**:  
   Execute the provided script, which will automatically download and place the dataset in the `data/` directory. Note that this requires a Kaggle account and API credentials (`~/.kaggle/kaggle.json`). Be prepared for a download size of approximately 23.43GB. 

2. **Manual Download**:  
   Navigate to the competition's [homepage](https://www.kaggle.com/competitions/birdclef-2024), and under the "Data" section, select the option to download the dataset. Once downloaded, extract the files into the `data/` directory to maintain compatibility with the provided scripts. (Kaggle might require an active account!)


The `data/unlabeled_soundscapes` directory is not used in this project and can be safely deleted to save storage space.

`ffmpeg` is the required backend of the `torchaudio` library.

The dependencies can be installed with the following commands (on Ubuntu):

```bash
sudo apt install ffmpeg
pip3 install -r requieremnts.txt
```

We are providing a way to dockerize the project, but since neither of us has any experience with docker we can't say we are fully confident in the process.

---

### Starting the Training

Training can be started after downloading the dataset and dependencies by running the `train_classifier.py` python file. The evaulation of the model is part of the script and is done automaticaly.

The output of the main script are the following:
- **`confusion_matrix.png`**: A confusion matrix generated by evaulating the model on the test dataset.
- **`roc_curve.png`**: A plot of the ROC curve generated by evaulating the model on the test dataset.
- **`models/model.pth`**: The final model.

Please note, that the training done with the attached config file takes up to 18 hours and with the slice mode on, the `processed_data` folder's size is over 203 GB.

We are planning the final verification of our model with the submission notebook `submission.ipynb`. 

---

### Configuration File

#### Key Sections in `config.ini`

- **`testing`**: Configures testing mode and data sample limits for debugging.
- **`callbacks`**: Manages logging options, such as Neptune.ai integration, model checkpointing and hyperopt mode.
- **`trainer`**: Sets training hyperparameters, including batch size, number of epochs, and random seed.
- **`agent`**: Specifies learning rate schedules, optimizer choices, and loss functions.
- **`data`**: Defines dataset split ratios, minimum samples per class, and shuffle options.
- **`data_process`**: Controls parameters for audio preprocessing, such as sample rate, FFT settings, and mel-spectrogram configurations.
- **`augmentation`**: Enables or disables data augmentation techniques, such as noise addition and spectrogram masking.

---

## Acknowledgments

This project was developed as part of the course "Deep Learning in Practice with Python and LUA" (BMEVITMAV45) at the Faculty of Electrical Engineering and Informatics, Budapest University of Technology and Economics.

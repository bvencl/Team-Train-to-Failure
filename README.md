# Team Train to Failure - [BirdCLEF 2024 @ Kaggle](https://www.kaggle.com/competitions/birdclef-2024)

## Developing Team - Train to Failure

 - Bódi Vencel (VBW5N9) - Villamosmérnöki és Informatikai Kar (Faculty of Electrical Engineering and Informatics)
 - Mitrenga Márk (OLLNTB) - Közlekedés és Járműmérnöki Kar (Faculty of Transportation Engineering and Vehicle Engineering)

"Train to Failure". Not because we aim to fail, but because, well... deep learning often has other plans.

## BirdCLEF competition

The **Kaggle BirdCLEF 2024** competition dares us to build machine learning models that can identify birds from audio recordings. No feathers or birdseed here, just the sweet sound of chirps, squawks, and whistles—perfectly suited foranyone who loves the idea of *birdsongs meeting backpropagation*.

The competition specifically focuses on Indian bird species from the Western Ghats, a biodiversity hotspot. So, while we’re tackling the code, we're also (at least in spirit) helping to protect endangered birds. 

To be precise: we’re diving into Passive Acoustic Monitoring (PAM), identifying bird species even at the eeriest hours. Our models will sift through hours of bird audio and spot those rare night owls and other lesser-known avian singers.

### Our ideas

We plan to train a Convolutional Neural Network (CNN) on the mel-spectrograms of the bird sounds. These spectrograms will be standardized per frequency to level the playing field—no cheating with loud squawks or whispers. This way, the CNN can focus on the actual patterns, not the volume wars. If our time allows, we might even pit this CNN against a Wave2Vec model to see who’s the real king of bird-spotting. 

### Downloading the Kaggle BirdCLEF-2024 dataset

Simply run the `download_data.sh` shell script to fetch the data into the **data/** folder. Warning: this is a Kaggle dataset, so you will need a Kaggle account (and patience, as it's around 23.43GB of birdsongs - you might want to listen to the script and grab a coffee or tea. Also, for now, we don't use the content provided in the `data/unlabeled_soundscapes`, so feel free to delete it.)

### Structure of the repository

 - **train_classifier.py**: The big one. Run this after setting up the dataset and installing the packages in `requirements.txt`. This will start our bird-identifying beast (at some point at least hopefully).
 - **dataloader/data_preprocess.py**: This script does the heavy lifting of loading, converting to mel-spectrograms, standardizing, and organizing metadata into a pandas DataFrame. It also saves the mel-spectograms in .npy files, to make further scripts simpler. 
 - **utils/visualise.py**: A little helper to visualize spectrograms and play the sounds. Handy for spotting noise or admiring the symphonies of our feathered friends before we mangle them into training data.
 - **utils/utils.py**: Well... We know no one likes the mysterious *"utils.py"* or *"utils.h"*, but there was no better place for functions so useful like these...
 - **utils/trainer.py**: This script contains the training loop, this is where the magic happens. Hopefully.
 - **factory/\***: This folder contains the factory scripts, which makes the repository much more readable.
 - **config.ini**: This `.ini` file contains the constants used for hyperparameters and other variables, which we want to separate for easy tuning.

### Config.ini

A quick look at the `config.ini` file:

 - **testing**:
   - **testing**: This should be interpreted as a boolean, decides wheter or not the testing mode is on. You might want to leave this on is you are only checking out if the script runs or not.
   - **data_samples_for_testing**: The number of samples used for testing the data pipeline and training loop.
 - **callbacks**:
   - **neptune_logger**: Interpreted as a boolean, which swithces on or off the use of a logger, which logs our data onto the Neptune.ai website.
   - **neptune_project**: The name for the Neptune project.
   - **neptune_token**: The secret Neptune token goes here.
   - **model_checkpoint**: Interpreted as a boolean, switches on and off the use of the model checkpoint implemented by us.
   - **model_checkpoint_type**: The metric the model checkpoint wants to optimize.
   - **model_checkpoint_verbose**: Decides if the model checkpoint logs onto the stdout or not.
 - **paths**: paths for the project
 - **trainer**:
   - **seed**: We use a global seed for everything.
   - **batch_size_*xyz***: batch size for *xyz* the model. We can use only small batches, because we have limited computing capacity.
   - **num_workers**: The number of processor cores dedicated to load the GPU.
   - **n_epochs**: Number of epochs for a training loop.
 - **agent**:
   - **lr_decay**: switches on and off the use of an lr scheduler.
   - **lr_decay_type**: with this you can switch between the lr schedulers that are implemented in our code.
   - **lr_start**: starting learning rate
   - **lr_warmup_end**: If you want to use the Warmup Cosine Annealing lr scheduler you have to specify the lr at the end of the warmup.
   - **lr_end**: in the same scenario as the **lr_warmup_end**, you will have to specify a learning rate for the end of the training.
   - **exp_gamma**: this constant varies the exponential decay if you want to use the exponential lr scheduler.
   - **lr_verbose**: makes the lr scheduler log the currently used lr onto the stdout.
   - **loss**: you can decide what lossfunction you want to use. Currently the focal loss does not work, but we are working on it.
   - **optimizer**: you can decide what optimizer to use, but we highly recommend adam
 - **model**:
   - **type**: you can choose a pretrained model form the implemented ones.
   - **transfer_learning**: decides if the model comes with a pretrained set of weigths or not
 - **data**:
   - **hash_path**: path for the hash made of the **data_process** config section to make sure that you dont have to preprocess the data every time you want to start a trining with the same preprocessing parameters.
   - **train_ratio**: Defines the ratio of the training dataset.
   - **test_val_ratio**: Defines the ratio of test adn validation datasets.
   - **min_samples_in_class**: There are classes with a sample number less than 10, we think that taking those into account only makes thing worse. This parameter defines the minimum samples a class need to have to take it into account.
   - **shuffle**: Decides if the trainig datset gets shuffled between epochs or not.
   - **output_dir**: the path for the output of the AudioPreprocesser class.
   - **num_workers**: CPU cores used paralell for preprocessing the data. Logic behind this is not implemented yet.
   - **multi_threading**: Decides if multi threading for the data preprocessing is enabled or not. Logic behind this is not implemented yet.
 - **data_process**:
   - **sample_rate**: The sample rate used for making the waveforms. This was given by the Kaggle team on the page of the competition.
   - **n_mels**: Number of discrete bins for the mel-spectograms.
   - **n_fft**: This defines the time window the STFT takes into account when its calculating the spectogram.
   - **hop_length**: The time step between STFT-s are taken.
   - **max_length_s**: The maximum length of the processed audio files in seconds.
   - **f_max**: Maximum frequency on the spectograms.
   - **f_min**: Minimum frequency on the spectograms.
   - **mode**: The mode we process the data. If the slice mode is on, and the processed audio file is longer than we want we chop the audio files **keeping** every segment of it. If the single mode is activated we throw away the rest of the audio file beside the start of it.
 - **augmentation**:
   - **data_augmentation**: Swithces on or off the data augmentation.
   - **augment_add_noise**: Adds noise to the images.
   - **augment_spec_augment**: Masks a random piece of the spectograms.

  

## Acknowledgements

The project was developed within the framework of the subject "Deep Learning in Practice with Python and LUA" (BMEVITMAV45) at the Faculty of Electrical Engineering and Informatics of Budapest University of Technology and Economics.

Generative AI was used for:
 - Code refactoring (**not writing code!** just keeping things neat)
 - Writing and refining comments
 - Maybe a little help with this readme

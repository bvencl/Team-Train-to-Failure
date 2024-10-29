# Team Train to Failure - [BirdCLEF 2024 @ Kaggle](https://www.kaggle.com/competitions/birdclef-2024)

## Developing Team - Train to Failure

 - Bódi Vencel (VBW5N9) - Villamosmérnöki és Informatikai Kar (Faculty of Electrical Engineering and Informatics)
 - Mitrenga Márk (OLLNTB) - Közlekedés és Járműmérnöki Kar (Faculty of Transportation Engineering and Vehicle Engineering)

Yes, you read it right: "Train to Failure". Not because we aim to fail, but because, well... deep learning often has other plans.

## BirdCLEF competition

The **Kaggle BirdCLEF 2024** competition dares us to build machine learning models that can identify birds from audio recordings. No feathers or birdseed here, just the sweet sound of chirps, squawks, and whistles—perfectly suited foranyone who loves the idea of *birdsongs meeting backpropagation*.

The competition specifically focuses on Indian bird species from the Western Ghats, a biodiversity hotspot. So, while we’re tackling the code, we're also (at least in spirit) helping to protect endangered birds. 

To be precise: we’re diving into Passive Acoustic Monitoring (PAM), identifying bird species even at the eeriest hours. Our models will sift through hours of bird audio and spot those rare night owls and other lesser-known avian singers.

### Our ideas

We plan to train a Convolutional Neural Network (CNN) on the mel-spectrograms of the bird sounds. These spectrograms will be standardized per frequency to level the playing field—no cheating with loud squawks or whispers. This way, the CNN can focus on the actual patterns, not the volume wars. If our time allows, we might even pit this CNN against a Wave2Vec model to see who’s the real king of bird-spotting. 

### Downloading the Kaggle BirdCLEF-2024 dataset

Simply run the `download_data.sh` shell script to fetch the data into the **data/** folder. Warning: this is a Kaggle dataset, so you will need a Kaggle account (and patience, as it's around 23.43GB of birdsongs - you might want to listen to the script and grab a coffe or tea).

### Structure of the repository

Here’s a quick look at the files that matter for this milestones:

 - **train_classifier.py**: The big one. Run this after setting up the dataset and installing the packages in `requirements.txt`. This will start our bird-identifying beast (at some point at least hopefully).
 - **dataloader/data_loader.py**: This script does the heavy lifting of loading, converting to mel-spectrograms, standardizing, and organizing metadata into a pandas DataFrame. At some point, this will probably evolve into a Torch DataLoader...
 - **utils/visualise.py**: A little helper to visualize spectrograms and play the sounds. Handy for spotting noise or admiring the symphonies of our feathered friends before we mangle them into training data.
 - **utils/utils.py**: Well... We know no one likes the mysterious *"utils.py"* or *"utils.h"*, but there was no better place for functions so useful like these...

## Acknowledgements

The project was developed within the framework of the subject "Deep Learning in Practice with Python and LUA" (BMEVITMAV45) at the Faculty of Electrical Engineering and Informatics of Budapest University of Technology and Economics.

Generative AI was (ethically!) employed for:
 - Code refactoring (**not writing code!** just keeping things neat)
 - Writing and refining comments (because clarity is key, even for scripts)
 - Maybe a little help with this readme (no one likes when their readme is full of typos)

from collections import Counter

from utils.utils import *
from utils.trainer import Trainer
from factory.dataloader_factory import DataLoaderFactory
from factory.callback_factory import CallbackFactory
from factory.agent_factory import AgentFactory
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from factory.transform_factory import TransformFactory
from data_loader.data_preprocess import AudioPreprocesser
from callback.visualiser import Visualiser
from utils.final_validation import final_validation
from utils.hyperopt import hyperopt_or_train


def main():

    # Read the configuration file and set the seeds
    args = get_args()
    config = read(args.config)
    seed = config.trainer.seed
    set_seeds(seed)

    # A simple tool for optional visualization of spectrograms
    visualiser = Visualiser(config=config)

    # Process the raw audio files, save them onto disk and make train, test and validation dataframes
    processer = AudioPreprocesser(config=config, visualiser=visualiser)
    train_df, val_df, test_df, label2idx, idx2label = processer.process_database()
    

    # Create the optional transforms 
    transforms = TransformFactory.create(config=config)
    # Make the custom datasets
    train_data, val_data, test_data, num_classes = DatasetFactory().create(
        config=config,
        transforms=transforms,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label2idx=label2idx,
        idx2label=idx2label,
    )

    # Logging
    num_classes = len(train_df["label"].unique())
    print(f"Number of classes: {num_classes} | Length of train data: {len(train_data)}")
    class_names = train_df['label'].unique().tolist
    
    # Create the Torch Dataloaders
    train_loader, val_loader, test_loader = DataLoaderFactory.create(
        config=config, train=train_data, val=val_data, test=test_data
    )

    hyperopt_or_train(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        num_classes=num_classes,
    )

if __name__ == "__main__":
    main()
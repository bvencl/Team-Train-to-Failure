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

    # Create the model, optimizer, lossfunction, learning rate scheduler and callbacks
    model = ModelFactory.create(config=config, num_classes=num_classes)
    lossfn, optimizer, lr_scheduler = AgentFactory.create(config=config, model=model)
    callbacks = CallbackFactory.create(
        config=config,
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        lossfn=lossfn,
    )
    # Trainer class for ease of use
    trainer = Trainer(
        config=config,
        criterion=lossfn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        callbacks=callbacks,
        model=model,
    )

    # Training loop
    model = trainer.train()

    # final validation of model
    final_validation(config=config, model=model, data_loader=val_loader if test_loader is None else test_loader, criterion=lossfn, num_classes=num_classes, class_names=class_names,  neptune_logger=callbacks["neptune_logger"] if config.callbacks.neptune_logger else None)

    # save the model weights
    torch.save(model.state_dict(), config.paths.model_path + config.paths.model_name)

    # upload the model weights to Neptune.ai
    if "neptune_logger" in callbacks and callbacks["neptune_logger"] is not None:
        callbacks["neptune_logger"].save_model(config.paths.model_path + config.paths.model_name)

if __name__ == "__main__":
    main()

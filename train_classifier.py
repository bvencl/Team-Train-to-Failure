from utils.utils import *
from utils.trainer import Trainer
from factory.callback_factory import CallbackFactory
from factory.dataloader_factory import DataLoaderFactory
from factory.agent_factory import AgentFactory
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from factory.transform_factory import TransformFactory
from data_loader.data_preprocess import AudioPreprocesser
from callback.visualiser import Visualiser


def main():
    args = get_args()
    config = read(args.config)
    seed = config.trainer.seed  # getting the seeds from the config
    set_seeds(seed)

    visualiser = Visualiser(config=config)

    processer = AudioPreprocesser(config=config, visualiser=visualiser)
    train_df, val_df, test_df = processer.process_database()

    visualiser(df_row=train_df.iloc[0])

    transforms = TransformFactory.create(config=config)

    train_data, val_data, test_data, num_classes = DatasetFactory().create(
        config=config,
        transforms=transforms,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    visualiser(y=train_data[0][0])

    print(f"Number of classes: {num_classes} | Length of train data: {len(train_data)}")

    train_loader, val_loader, test_loader = DataLoaderFactory.create(
        config=config, train=train_data, val=val_data, test=test_data
    )

    model = ModelFactory.create(config=config, num_classes=num_classes)
    lossfn, optimizer, lr_scheduler = AgentFactory.create(config=config, model=model)
    callbacks = CallbackFactory.create(
        config=config,
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        lossfn=lossfn,
    )
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
    trainer.train()


if __name__ == "__main__":
    main()

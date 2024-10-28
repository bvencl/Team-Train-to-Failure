from utils.data_loader import *
from utils.utils import *
from utils.trainer import Trainer
from factory.callback_factory import CallbackFactory
from factory.dataloader_factory import DataLoaderFactory
from factory.agent_factory import AgentFactory
from factory.model_factory import ModelFactory


def main():

    args = get_args()
    config = read(args.config)
    seed = config.trainer.seed
    set_seeds(seed)

    df_train, df_val, df_test = load_data(config)

    train_loader, val_loader, test_loader = DataLoaderFactory.create(
        config=config, train=df_train, val=df_val, test=df_test
    )
    model = ModelFactory.create(
        config=config, num_classes=df_train["primary_label"].nunique()
    )
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

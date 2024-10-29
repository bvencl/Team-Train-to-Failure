from utils.utils import *
from utils.trainer import Trainer
from factory.callback_factory import CallbackFactory
from factory.dataloader_factory import DataLoaderFactory
from factory.agent_factory import AgentFactory
from factory.model_factory import ModelFactory
from utils.visualizer import Visualizer


from data_loader.data_loader import *

"""We commented only those files that are relevant for the current (first) milestone"""

def main():
    args = get_args()
    config = read(args.config)
    seed = config.trainer.seed # getting the seeds from the config
    set_seeds(seed)

    df_train, df_val, df_test = load_data(config)

    visualizer = Visualizer()
    visualizer.show_and_play(df_train)

    # !The following are not relevant for the first milestone!
    if True: 
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

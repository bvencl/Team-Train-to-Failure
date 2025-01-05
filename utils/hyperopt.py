import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from factory.model_factory import ModelFactory
from factory.callback_factory import CallbackFactory
from factory.agent_factory import AgentFactory
from utils.trainer import Trainer
from utils.final_validation import final_validation


class Hyperopt:
    def __init__(self, config, train_loader, val_loader, test_loader, class_names, num_classes):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.num_classes = num_classes
        self.hyperopt_on = config.callbacks.hyperopt
        
    def hyperopt_or_train(self):
        model = ModelFactory.create(config=self.config, num_classes=self.num_classes)
        if self.config.callbacks.hyperopt:
            # Define the hyperparameter search space
            hyperopt_config = {
                "lr_decay_type": tune.choice(["lin", "exp", "cos", "warmup_cos"]),
                "lr_start": tune.loguniform(1e-7, 1e-3),
                "lr_warmup_end": tune.loguniform(1e-7, 1e-3),
                "lr_end": tune.loguniform(1e-7, 1e-3),
                "warmup_epochs": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "exp_gamma": tune.uniform(0.9, 0.999),
                "loss": tune.choice(["cross_entropy", "focal_loss"]),
                "optimizer": tune.choice(["sgd", "adam","rmsprop", "adamw"]),
            }

            # Asynchronous Hyperband Scheduler
            hyperopt_scheduler = ASHAScheduler(
                # metric="accuracy",
                # mode="max",
                max_t=10,
                grace_period=1,
                reduction_factor=2
            )

            # Begin hyperparameter search
            analysis = tune.run(
                tune.with_parameters(
                    self.train_model,
                    model=model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    test_loader=self.test_loader,
                    class_names=self.class_names,
                    num_classes=self.num_classes,
                ),
                resources_per_trial={"cpu": 6, "gpu": 0.95},
                config=hyperopt_config,
                scheduler=hyperopt_scheduler,
                num_samples=100,  # Number of hyperparameter combinations to try
                metric="accuracy",
                mode="max"
            )

            print("Best hyperparameters found were: ", analysis.best_config)
            
            df = analysis.results_df

            # Eredmények írása CSV-be
            df.to_csv("tune_results.csv", index=False)
            print("Eredmények mentve a tune_results.csv fájlba!")
            
        else:
            self.train_model(self.config, model, self.train_loader, self.val_loader, self.test_loader, self.class_names, self.num_classes)



    def train_model(self, config, model=None, train_loader=None, val_loader=None, test_loader=None, class_names=None, num_classes=None):
        # Create the model, optimizer, lossfunction, learning rate scheduler and callbacks
        lossfn, optimizer, lr_scheduler = AgentFactory.create(original_config=self.config, hyperopt_on=self.hyperopt_on, config=config, model=model)
        callbacks = CallbackFactory.create(
            config=self.config,
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            lossfn=lossfn,
        )
        # Trainer class for ease of use
        trainer = Trainer(
            config=self.config,
            criterion=lossfn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr_scheduler=lr_scheduler,
            callbacks=callbacks,
            model=model,
        )

        # Kiírja a jelenleg használt konfigurációt
        # print("\n=== Jelenlegi paraméterek ===")
        # for key, value in config.items():
        #     print(f"{key}: {value}")
        # print("=============================\n")

        # Training loop
        model = trainer.train()

        # final validation of model
        final_validation(
            config=config,
            model=model,
            data_loader=test_loader,
            criterion=lossfn,
            num_classes=num_classes,
            class_names=class_names,
            neptune_logger=callbacks["neptune_logger"] if config.callbacks.neptune_logger else None
        )

        # save the model weights
        torch.save(model.state_dict(), config.paths.model_path + config.paths.model_name)

        # upload the model weights to Neptune.ai
        if "neptune_logger" in callbacks and callbacks["neptune_logger"] is not None:
            callbacks["neptune_logger"].save_model(config.paths.model_path + config.paths.model_name)
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
        """
        Initializes the Hyperopt class with configuration, data loaders, and class information.

        Args:
            config: Configuration object containing hyperparameter search and training settings.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            test_loader: DataLoader for the test dataset.
            class_names: List of class names.
            num_classes: Number of classes in the dataset.
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.num_classes = num_classes
        self.hyperopt_on = config.callbacks.hyperopt

    def hyperopt_or_train(self):
        """
        Decides whether to perform hyperparameter optimization or directly train the model.
        """
        # Create the model using the specified configuration.
        model = ModelFactory.create(config=self.config, num_classes=self.num_classes)
        
        if self.config.callbacks.hyperopt:  # Perform hyperparameter optimization if enabled.
            # Define the hyperparameter search space.
            hyperopt_config = {
                "lr_decay_type": tune.choice(["lin", "exp", "cos", "warmup_cos"]),
                "lr_start": tune.loguniform(1e-6, 1e-2),
                "lr_warmup_end": tune.loguniform(1e-4, 1e-2),
                "lr_end": tune.loguniform(1e-8, 1e-4),
                "warmup_epochs": tune.choice([1, 3, 5, 7, 10, 15, 20]),
                "exp_gamma": tune.uniform(0.85, 0.995),
                "loss": tune.choice(["cross_entropy", "focal_loss"]),
                "optimizer": tune.choice(["sgd", "adam", "adamw", "rmsprop"]),
            }

            # Configure the ASHA scheduler for efficient hyperparameter optimization.
            hyperopt_scheduler = ASHAScheduler(
                max_t=10,  # Maximum training epochs per trial.
                grace_period=1,  # Minimum epochs before early stopping.
                reduction_factor=2  # Factor by which trials are reduced.
            )

            # Run the hyperparameter search.
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
                resources_per_trial={"cpu": 4, "gpu": 0.5},  # Resource allocation for each trial.
                config=hyperopt_config,  # Hyperparameter search space.
                scheduler=hyperopt_scheduler,  # Scheduler to manage trial execution.
                num_samples=100,  # Number of trials to run.
                metric="accuracy",  # Evaluation metric to optimize.
                mode="max"  # Optimize for maximum accuracy.
            )

            # Output the best hyperparameters found.
            print("Best hyperparameters found were: ", analysis.best_config)

            # Save results to a CSV file for further analysis.
            df = analysis.results_df
            df.to_csv("tune_results.csv", index=False)
            print("Results saved to tune_results.csv!")
        
        else:  # Train the model directly if hyperparameter optimization is disabled.
            self.train_model(self.config, model, self.train_loader, self.val_loader, self.test_loader, self.class_names, self.num_classes)

    def train_model(self, config, model=None, train_loader=None, val_loader=None, test_loader=None, class_names=None, num_classes=None):
        """
        Trains the model using the specified configuration and data loaders.

        Args:
            config: Configuration object.
            model: PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            test_loader: DataLoader for test data.
            class_names: List of class names.
            num_classes: Number of classes in the dataset.
        """
        # Create the loss function, optimizer, and learning rate scheduler.
        lossfn, optimizer, lr_scheduler = AgentFactory.create(original_config=self.config, hyperopt_on=self.hyperopt_on, config=config, model=model)
        
        # Create callbacks for logging and checkpointing.
        callbacks = CallbackFactory.create(
            config=self.config,
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            lossfn=lossfn,
        )

        # Initialize the trainer with the specified parameters.
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

        # Train the model.
        model = trainer.train()

        # Perform final validation on the test dataset.
        final_validation(
            config=config,
            model=model,
            data_loader=test_loader if test_loader is not None else val_loader,
            criterion=lossfn,
            num_classes=num_classes,
            class_names=class_names,
            neptune_logger=callbacks["neptune_logger"] if config.callbacks.neptune_logger else None
        )

        # Save the trained model to disk.
        torch.save(model.state_dict(), config.paths.model_path + config.paths.model_name)

        # Optionally upload the trained model to Neptune.ai if logging is enabled.
        if "neptune_logger" in callbacks and callbacks["neptune_logger"] is not None:
            callbacks["neptune_logger"].save_model(config.paths.model_path + config.paths.model_name)
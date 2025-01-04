from ray import tune
from ray.tune.schedulers import ASHAScheduler

def hyperopt(trainer):
    train_model = trainer.train()

    # Define the hyperparameter search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8])
    }

    # Asynchronous Hyperband Scheduler
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    # Begin hyperparameter search
    analysis = tune.run(
        train_model,
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        config=config,
        scheduler=scheduler,
        num_samples=10  # Number of hyperparameter combinations to try
    )

    print("Best hyperparameters found were: ", analysis.best_config)
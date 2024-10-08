import neptune
from neptune_pytorch import NeptuneLogger


class CustomNeptuneLogger(NeptuneLogger):
    def __init__(self, token, project, config):
        self.run = neptune.init_run(api_token=token, project=project)
        self._config = config
        self.logger = None

    def start_logging(self, model):
        params = {
            "batch_size_train": self._config.data_loader.batch_size_train,
            "num_workers": self._config.data_leader.num_workers,
            "seed": self._config.utils,
            "starting_learning_rate": self._config.agent.starting_learing_rate,
            "lr_min": self._config.agent.lr_min,
            "lr_max_at_end": self._config.agent.lr_max_at_end,
            "lr_decay_type": self._config.agent.lr_decay_type,
            "warmup_epochs": self._config.agent.warmup_epochs,
            "warmup_lr_high": self._config.agent.warmup_lr_high,
            "loss": self._config.agent.loss,
            "optimizer": self._config.agent.optimizer,
        }

        self.run["parameters"] = params
    
        self.logger = NeptuneLogger(run=self.run, model=model)

    def save_model(self):
        self.run["checkpoint_models"].upload("model_checkpoint/checkpoint.pth")

    def stop(self):
        self.run.stop()

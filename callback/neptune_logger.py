import neptune
from neptune_pytorch import NeptuneLogger


class CustomNeptuneLogger(NeptuneLogger):
    def __init__(self, token, project, config):
        self.run = neptune.init_run(api_token=token, project=project)
        self._config = config
        self.logger = None

    def start_logging(self, model):
        """
        Starts logging parameters, data, and configurations to Neptune.
        Logs all relevant parameters from the config.
        """
        parameters = {}

        # General logging function
        def add_param(key, value):
            parameters[key] = value

        # Trainer-related parameters
        add_param("seed", self._config.trainer.seed)
        add_param("n_epochs", self._config.trainer.n_epochs)
        add_param("batch_size_train", self._config.trainer.batch_size_train)
        add_param("batch_size_val", self._config.trainer.batch_size_val)
        add_param("batch_size_test", self._config.trainer.batch_size_test)
        add_param("num_workers", self._config.trainer.num_workers)

        # Agent-related parameters
        add_param("optimizer", self._config.agent.optimizer)
        add_param("lr_start", self._config.agent.lr_start)
        add_param("loss", self._config.agent.loss)
        if self._config.agent.lr_decay:
            add_param("lr_decay", self._config.agent.lr_decay)
            add_param("lr_decay_type", self._config.agent.lr_decay_type)
            add_param("lr_end", self._config.agent.lr_end)

            if self._config.agent.lr_decay_type == "exp":
                add_param("exp_gamma", self._config.agent.exp_gamma)
            if self._config.agent.lr_decay_type == "warmupcosine":
                add_param("lr_warmup_end", self._config.agent.lr_warmup_end) 

        # Model-related parameters
        add_param("model_type", self._config.model.type)
        add_param("transfer_learning", self._config.model.transfer_learning)

        # Data-related parameters
        add_param("train_ratio", self._config.data.train_ratio)
        add_param("test_val_ratio", self._config.data.test_val_ratio)
        add_param("shuffle", self._config.data.shuffle)
        add_param("min_samples_in_class", self._config.data.min_samples_in_class)

        # Data process parameters
        add_param("sample_rate", self._config.data_process.sample_rate)
        add_param("n_mels", self._config.data_process.n_mels)
        add_param("n_fft", self._config.data_process.n_fft)
        add_param("hop_length", self._config.data_process.hop_length)
        add_param("max_length_s", self._config.data_process.max_length_s)
        add_param("f_max", self._config.data_process.f_max)
        add_param("f_min", self._config.data_process.f_min)
        add_param("mode", self._config.data_process.mode)

        # Augmentation parameters
        add_param("data_augmentation", self._config.augmentation.data_augmentation)
        add_param("augment_add_noise", self._config.augmentation.augment_add_noise)
        add_param("augment_spec_augment", self._config.augmentation.augment_spec_augment)

        # Callback parameters
        add_param("neptune_logger", self._config.callbacks.neptune_logger)
        add_param("model_checkpoint", self._config.callbacks.model_checkpoint)
        add_param("model_checkpoint_type", self._config.callbacks.model_checkpoint_type)
        add_param("model_checkpoint_verbose", self._config.callbacks.model_checkpoint_verbose)

        # Additional parameters
        add_param("dataset", self._config.data.output_dir)
        add_param("output_dir", self._config.data.output_dir)

        # Normalize all parameter values (replace None with default values)
        parameters = {k: (v if v is not None else "N/A") for k, v in parameters.items()}

        # Log parameters to Neptune
        self.run["parameters"] = parameters
        self.logger = NeptuneLogger(run=self.run, model=model)

    def save_model(self, path):
        self.run["model"].upload(path)

    def stop(self):
        self.run.stop()

    def __del__(self):
        pass

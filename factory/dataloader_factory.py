import torch
from torch.utils.data import DataLoader
from factory.base_factory import BaseFactory


class DataLoaderFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs):
        train_dataset, val_dataset, test_dataset = kwargs["train"], kwargs["val"], kwargs["test"]
        config = kwargs["config"]

        shuffle = config.data.shuffle
        batch_size = config.trainer.batch_size_train
        device = "cuda" if torch.cuda.is_available() else ""

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=config.trainer.num_workers,
            shuffle=shuffle,
            pin_memory=True if device != "" else False,
            pin_memory_device=device,
        )
        print(f'Number of train batches: {len(train_loader)}')
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.trainer.batch_size_val,
            num_workers=config.trainer.num_workers,
            pin_memory=True if device != "" else False,
            pin_memory_device=device,
        )
        print(f'Number of validation batches: {len(val_loader)}')

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.trainer.batch_size_test,
            num_workers=config.trainer.num_workers,
            pin_memory=True if device != "" else False,
            pin_memory_device=device,
        )
        print(f'Number of test batches: {len(test_loader)}')

        return train_loader, val_loader, test_loader

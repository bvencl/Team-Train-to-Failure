from factory.base_factory import BaseFactory
from data_loader.custom_dataset import BirdClefDataset
from data_loader.load_data import load_birdclef


class DatasetFactory(BaseFactory):

    @classmethod
    def create(cls, **kwargs):
        config = kwargs["config"]

        birdclef_data = load_birdclef(config)

        train_dataset = BirdClefDataset(
            data=birdclef_data["train"]["data"],
            labels=birdclef_data["train"]["labels"],
            interpretable_labels=birdclef_data["train"]["common_name"],
            position=birdclef_data["train"]["position"],
            files=birdclef_data["train"]["files"],
            transform=birdclef_data["train"]["transforms"],
        )
        val_dataset = BirdClefDataset(
            data=birdclef_data["val"]["data"],
            labels=birdclef_data["val"]["labels"],
            interpretable_labels=birdclef_data["val"]["common_name"],
            position=birdclef_data["val"]["position"],
            files=birdclef_data["val"]["files"],
            transform=birdclef_data["val"]["transforms"],
        )
        test_dataset = BirdClefDataset(
            data=birdclef_data["test"]["data"],
            labels=birdclef_data["test"]["labels"],
            interpretable_labels=birdclef_data["test"]["common_name"],
            position=birdclef_data["test"]["position"],
            files=birdclef_data["test"]["files"],
            transform=birdclef_data["test"]["transforms"],
        )
        
        num_classes = birdclef_data["train"]["labels"][0].shape[0]

        return train_dataset, val_dataset, test_dataset, num_classes

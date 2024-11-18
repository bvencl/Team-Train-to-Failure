from factory.base_factory import BaseFactory
from data_loader.custom_dataset import BirdClefDataset
from data_loader.load_metadata import load_metadata


class DatasetFactory(BaseFactory):
    """
    Factory class to create training, validation, and test datasets
    based on the BirdClefDataset and metadata.
    """

    @classmethod
    def create(cls, config, transforms=None, train_df=None, val_df=None, test_df=None):
        """
        Creates datasets for training, validation, and testing.

        Args:
            config: Configuration object with paths and data-related parameters.
            transforms: Optional dictionary containing transforms for train, val, and test datasets.

        Returns:
            train_dataset, val_dataset, test_dataset, num_classes
        """
        

        # Optinal transformations made by the TransformFactory
        

        # Create the datasets
        train_dataset = BirdClefDataset(
            metadata_df=train_df,
            config=config,
            transform=transforms,
        )
        val_dataset = BirdClefDataset(
            metadata_df=val_df,
            config=config,
            # transform=val_transform,
        )
        test_dataset = BirdClefDataset(
            metadata_df=test_df,
            config=config,
            # transform=test_transform,
        )

        # Number of classes
        num_classes = len(train_df["label"].unique())

        return train_dataset, val_dataset, test_dataset, num_classes

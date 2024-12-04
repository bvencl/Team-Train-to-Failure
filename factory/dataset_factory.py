from factory.base_factory import BaseFactory
from data_loader.custom_dataset import BirdClefDataset


class DatasetFactory(BaseFactory):
    """
    Factory class to create training, validation, and test datasets
    based on the BirdClefDataset and metadata.
    """

    @classmethod
    def create(cls, config, transforms=None, train_df=None, val_df=None, test_df=None, label2idx=None, idx2label=None):
        """
        Creates *BirdClefDatasets* for training, validation, and testing.

        Args:
            - **config** (Configuration object with paths and data-related parameters.)
            - **transforms** (Optional dictionary containing transforms for train, val, and test datasets.)
            - **train_df** (DataFrame containing metadata for training data.)
            - **val_df** (DataFrame containing metadata for validation data.)
            - **test_df** (DataFrame containing metadata for test data.)
            - **label2idx** (Dictionary mapping labels to indices.)
            - **idx2label** (Dictionary mapping indices to labels.)

        Returns:
            - **train_dataset** (BirdClefDataset object for training data.)
            - **val_dataset** (BirdClefDataset object for validation data.)
            - **test_dataset** (BirdClefDataset object for test data.)
            - **num_classes** (Number of classes in the dataset.)
        """        

        # Create the datasets
        train_dataset = BirdClefDataset(
            metadata_df=train_df,
            config=config,
            transform=transforms,
            label2idx=label2idx,
            idx2label=idx2label,
        )
        val_dataset = BirdClefDataset(
            metadata_df=val_df,
            config=config,
            label2idx=label2idx,
            idx2label=idx2label,
        )
        test_dataset = BirdClefDataset(
            metadata_df=test_df,
            config=config,
            label2idx=label2idx,
            idx2label=idx2label,
        )

        # Number of classes
        num_classes = len(train_df["label"].unique())

        return train_dataset, val_dataset, test_dataset, num_classes

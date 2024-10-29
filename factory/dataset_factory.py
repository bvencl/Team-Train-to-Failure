from factory.base_factory import BaseFactory
from data_loader.data_loader import load_data
class DatasetFactory(BaseFactory):
    
    @classmethod
    def create(cls, **kwargs):
        config = kwargs["config"]

        df_train, df_val, df_test = load_data(config)

        train_data = df_train["mel_spectogram"]
        train_labels = df_train["one_hot_vector"]
        train_interpretable_labels = df_train["common_name"]
        train_position = (df_train["latitude"], df_train["longitude"])
        
        val_data = df_val["mel_spectogram"]
        val_labels = df_val["one_hot_vector"]
        val_interpretable_labels = df_val["common_name"]
        val_position = (df_val["latitude"], df_val["longitude"])

        test_data = df_test["mel_spectogram"]
        test_labels = df_test["one_hot_vector"]
        test_interpretable_labels = df_test["common_name"]
        test_position = (df_test["latitude"], df_test["longitude"])

        
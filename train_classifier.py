from utils.data_loader import *
from utils.utils import *
# from utils.validate_model import validate_model
from factory.callback_factory import CallbackFactory
from factory.dataloader_factory import DataLoaderFactory
def main():

    args = get_args()
    config = read(args.config)
    seed = config.utils.seed
    set_seeds(seed)


    df_train, df_val, df_test = load_data()

    train_loader, val_loader, test_loader = DataLoaderFactory(config, df_train, df_val, df_test)

    callbacks = CallbackFactory(config, model, val_loader, test_loader, lossfn)

if __name__ == "__main__":
    main()

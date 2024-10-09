from utils.data_loader import *
from utils.utils import *
from factory.callback_factory import CallbackFactory

def main():

    args = get_args()
    config = read(args.config)
    seed = config.utils.seed
    set_seeds(seed)


    df_train, df_val, df_test = load_data()


    callbacks = CallbackFactory(config, model, val_loader, lossfn)

if __name__ == "__main__":
    main()

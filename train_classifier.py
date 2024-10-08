from utils.data_loader import *
from utils.utils import *
import pandas as pd


def main():

    args = get_args()
    config = read(args.config)
    seed = config.utils.seed
    set_seeds(seed)

    df_train, df_val, df_test = load_data()


if __name__ == "__main__":
    main()

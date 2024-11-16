import pandas as pd
import os

def load_metadata(csv_path, min_samples_per_class=10):
    # Csak a szükséges oszlopokat töltjük be
    columns_to_read = [
        "primary_label", "type", "latitude", "longitude", "common_name", "filename"
    ]
    df = pd.read_csv(csv_path, usecols=columns_to_read)
    
    # Az osztályok szűrése: csak azok az osztályok maradnak, amelyeknek elég mintája van
    df = df[df.groupby("primary_label")["primary_label"].transform("count") >= min_samples_per_class]
    
    # Ellenőrizzük, hogy a fájlok elérhetőek-e
    df["file_path"] = df["filename"].apply(lambda x: os.path.join("data/train_audio", x))
    df = df[df["file_path"].apply(os.path.exists)]
    
    return df
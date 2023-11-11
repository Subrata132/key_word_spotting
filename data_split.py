import os
import numpy as np
import pandas as pd


def split_data():
    data_dir = "./speech_commands_v0.02"
    folders = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    class_to_idx = {}
    for i, folder in enumerate(folders):
        class_to_idx[folder] = i

    data_df = pd.DataFrame(columns=["folder", "filename", "class_idx"])
    for folder in folders:
        df = pd.DataFrame()
        filenames = os.listdir(os.path.join(data_dir, folder))
        class_idx = class_to_idx[folder]
        df["folder"] = [folder]*len(filenames)
        df["filename"] = filenames
        df["class_idx"] = [class_idx]*len(filenames)
        data_df = pd.concat([data_df, df], ignore_index=True)

    data_df = data_df.sample(frac=1).reset_index(drop=True)
    train_ = int(0.7*data_df.shape[0])
    val_ = int(0.8*data_df.shape[0])
    train_df = data_df[:train_].reset_index(drop=True)
    val_df = data_df[train_:val_].reset_index(drop=True)
    test_df = data_df[val_:].reset_index(drop=True)
    data_df.to_csv("./features/all_data.csv", index=False)
    train_df.to_csv("./features/sa_train.csv", index=False)
    val_df.to_csv("./features/sa_val.csv", index=False)
    test_df.to_csv("./features/sa_test.csv", index=False)
    print(f'Train split: {train_df.shape}')
    print(f'Validation split: {val_df.shape}')
    print(f'Test split: {test_df.shape}')


if __name__ == '__main__':
    split_data()


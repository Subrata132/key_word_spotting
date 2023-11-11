import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomLoader(Dataset):
    def __init__(self, feat_path, csv_path):
        self.feat_path = feat_path
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.info_list = df.values.tolist()
        del df

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, item):
        folder, name, label = self.info_list[item]
        feat_loc = os.path.join(self.feat_path, str(folder), '{}.npy'.format(name.split(".")[0]))
        feat = np.load(feat_loc)
        return feat, label


def load_data(batch_size, feat_path, csv_path, shuffle):
    dataset = CustomLoader(
        feat_path=feat_path,
        csv_path=csv_path
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader


if __name__ == '__main__':
    data_loader = load_data(
        batch_size=16,
        feat_path="./features",
        csv_path="./features/sa_val.csv",
        shuffle=True
    )
    for feat, label in data_loader:
        print(f'Feature shape: {feat.shape} | Label shape: {label.shape}')
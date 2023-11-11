import os
import itertools
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
import librosa
from tqdm import tqdm
from utils import create_folder, trim_or_pad_audio


class ExtractFeature:
    def __init__(
            self,
            fs=16000,
            nfft=128,
            hop_len=64,
            win_len=128,
            nb_mel_bins=251,
            data_loc="./speech_commands_v0.02",
            save_loc="./features_new",
            all_csv_loc="./features/all_data.csv",
            train_csv_loc="./features/sa_train.csv",
            window="hann"
    ):
        self.fs = fs
        self.nfft = nfft
        self.hop_len = hop_len
        self.win_len = win_len
        self.nb_mel_bins = nb_mel_bins
        self.data_loc = data_loc
        self.save_loc = save_loc
        self.window = window
        self.eps = 1e-10
        self.all_data_df = pd.read_csv(all_csv_loc)
        self.train_df = pd.read_csv(train_csv_loc)
        self.mel_wts = librosa.filters.mel(
            sr=self.fs,
            n_fft=self.nfft,
            n_mels=self.nb_mel_bins
        )
        create_folder(self.save_loc)

    def get_spectrogram(self, audio_path):
        audio, fs = librosa.load(audio_path, sr=None)
        audio = (audio - audio.mean()) / (audio.std() + self.eps)
        audio = trim_or_pad_audio(audio, t=1, fs=self.fs)
        stft = librosa.core.stft(
            np.asfortranarray(audio),
            n_fft=self.nfft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window
        )
        #         spectrogram = librosa.amplitude_to_db(np.abs(stft))
        mag_spectra = np.abs(stft) ** 2
        #         print(mag_spectra.shape, self.mel_wts.shape)
        spectrogram = np.dot(mag_spectra, self.mel_wts)
        #         spectrogram = librosa.feature.melspectrogram(S=mag_spectra, sr=self.fs)
        return spectrogram

    def generate_features(self):
        folders = self.all_data_df["folder"].unique()
        for folder in folders:
            folder_path = os.path.join(self.save_loc, str(folder))
            create_folder(folder_path)
            folder_df = self.all_data_df[self.all_data_df["folder"] == folder]
            with tqdm(total=folder_df.shape[0], desc=f'Folder: {folder}') as pbar:
                for _, row in folder_df.iterrows():
                    pbar.update(1)
                    audio_path = os.path.join(self.data_loc, str(folder), row["filename"])
                    spectrogram = self.get_spectrogram(audio_path=audio_path)
                    feat_path = os.path.join(folder_path, '{}.npy'.format(row["filename"].split(".")[0]))
                    np.save(feat_path, spectrogram)

    def normalize_feature(self):
        normalized_features_wts_file = os.path.join(self.save_loc, "spec_scaler")
        spec_scaler = preprocessing.StandardScaler()
        with tqdm(total=self.train_df.shape[0], desc="Fitting Scaler: ") as pbar:
            for _, row in self.train_df.iterrows():
                pbar.update(1)
                feat_path = os.path.join(self.save_loc, str(row["folder"]), '{}.npy'.format(row["filename"].split(".")[0]))
                feat_file = np.load(feat_path)
                spec_scaler.partial_fit(feat_file)
                del feat_file
        joblib.dump(
            spec_scaler,
            normalized_features_wts_file
        )
        with tqdm(total=self.all_data_df.shape[0], desc="Normalizing Features: ") as pbar:
            for _, row in self.all_data_df.iterrows():
                pbar.update(1)
                feat_path = os.path.join(self.save_loc, str(row["folder"]), '{}.npy'.format(row["filename"].split(".")[0]))
                feat_file = np.load(feat_path)
                feat_file = spec_scaler.transform(feat_file)
                np.save(feat_path, np.expand_dims(feat_file, axis=0))
                del feat_file


if __name__ == '__main__':
    extractor = ExtractFeature()
    extractor.generate_features()
    extractor.normalize_feature()

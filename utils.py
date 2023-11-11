import os
import math
import numpy as np


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print("Folders created")
    else:
        print("Folder already exists!")


def trim_or_pad_audio(audio, t=6, fs=44100):
    max_len = t*fs
    shape = audio.shape
    if shape[0] >= max_len:
        audio = audio[:max_len]
    else:
        n_pad = max_len - shape[0]
        zero_shape = (n_pad,)
        audio = np.concatenate((audio, np.zeros(zero_shape)), axis=0)
    return audio


def gpu_to_cpu(x):
    return x.clone().detach().cpu().numpy()
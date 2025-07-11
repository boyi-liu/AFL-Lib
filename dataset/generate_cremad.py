"client num < 30 is appropriate, otherwise the data is too small"

import os
import random
from pathlib import Path

import cv2 as cv
import librosa
import numpy as np
import torch
import yaml
from scipy import signal
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from utils.dataset_utils import check, save_file, separate_data, split_data



random.seed(1)
np.random.seed(1)

EMOTION_LABELS = ['NEU', 'HAP', 'SAD', 'FEA', 'DIS', 'ANG']


def generate_dataset(cfg):
    dir_path = cfg['dir_path']
    os.makedirs(dir_path, exist_ok=True)
    if check(cfg): return

    raw = f"{dir_path}/"
    if not os.path.exists(raw):
        raise RuntimeError("Please decompose the raw data of CREMA-D to /CREMA-D/")

    X, y = [], []

    img_transforms_cremad = v2.Compose([
        v2.ToImage(),
        v2.Resize((672, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    cremad_dataset = CREMAD_Dataset(root=raw, transform=img_transforms_cremad)

    dataloader = torch.utils.data.DataLoader(cremad_dataset, batch_size=1, shuffle=False)

    for image, audio, label in tqdm(dataloader, total=len(dataloader)):
        # X[0].append(image.squeeze(0).numpy())
        # X[1].append(audio.squeeze(0).numpy())
        X.append([image.squeeze(0).numpy(), audio.squeeze(0).numpy()])
        y.extend(label)

    cfg['class_num'] = len(set(y))
    X = np.array(X, dtype=object)
    y = torch.tensor(y).numpy()
    X, y, statistic = separate_data((X, y), cfg)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)


class CREMAD_Dataset(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.sr = 16000

        self.video_dir = Path(root) / 'VideoFlash'
        self.audio_dir = Path(root) / 'AudioWAV'
        self.full_data = os.listdir(self.video_dir)
        self.labels = ['']
        sorted(self.full_data)

    def __len__(self):
        return len(self.full_data)

    def __getitem__(self, index):
        vid_path = self.video_dir / self.full_data[index]
        file = Path(vid_path).name
        filename = Path(vid_path).stem
        audio_path = self.audio_dir / (filename + '.wav')

        # === get label from filename ===
        label = EMOTION_LABELS.index(filename.split('_')[2])
        # label = torch.tensor(label)
        if self.target_transform:
            label = self.target_transform(label)

        # === read video frames ===
        cap = cv.VideoCapture(vid_path)
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, total - 1, 5, dtype=int)[1:-1]
        frames = []
        for i in idxs:
            cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ret, f = cap.read()
            if not ret:
                continue
            f = cv.cvtColor(f, cv.COLOR_BGR2RGB)
            frames.append(f)
        cap.release()

        frame_out = np.concatenate(frames, axis=0)

        if self.transform:
            frame_out = self.transform(frame_out)

        # === extract audio feature ===
        samples, rate = librosa.load(audio_path, sr=self.sr)
        resamples = np.tile(samples, 3)[:self.sr * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        frequencies, times, spectrogram = signal.spectrogram(resamples, rate, nperseg=512, noverlap=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        audio_tensor = torch.tensor(spectrogram).unsqueeze(0)

        return frame_out, audio_tensor, label


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)

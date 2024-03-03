import torch
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from torchvision import datasets, transforms
import librosa
import soundfile

default_mfcc_params = {
    "n_fft": 2048,
    "win_length": None,
    "n_mels": 256,
    "hop_length": 512,
    "htk": True,
    "norm_melspec": None,
    "n_mfcc": 256,
    "norm_mfcc": "ortho",
    "dct_type": 2,
}



def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:

    classes = sorted(item.name for item in os.scandir(
        directory) if item.is_dir())

    class_to_idx = {classname: i for i, classname in enumerate(classes)}

    return classes, class_to_idx


class CustomAudioDataset(Dataset):

    def __init__(self, target_dir: str, transform=None, mfcc_params: Dict = default_mfcc_params, sr: int = 16000) -> None:
        self.paths = list(Path(target_dir).glob("*/*.wav"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)
        self.sr = sr
        self.mfcc_params = mfcc_params

    def load_audio(self, index: int) -> Tuple[np.ndarray, int]:
        audio_path = self.paths[index]
        audio, sr = librosa.load(audio_path, sr=self.sr)
        return audio, sr

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        audio, sr = self.load_audio(index)
        audio = mfcc_transform(y=audio,
                               sr=sr,
                               mfcc_params=self.mfcc_params)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(audio), class_idx
        else:
            return audio, class_idx

    def get_filepath(self, idx: int) -> Path:
        return self.paths[idx]

# takes tensor and output file paths as input and returns the path of the output file in str format


def convert_tensor_to_audio(y: torch.Tensor, filename: str | Path, output_dir: str, sr: int = 16000) -> str:
    output_file_path = Path(output_dir) / Path(filename)
    y = y.numpy()
    soundfile.write(output_file_path, y=y, sr=sr)
    return str(output_file_path)


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = 1):

    train_dataset = CustomAudioDataset(target_dir=train_dir,
                                       transform=transform,
                                       )
    test_dataset = CustomAudioDataset(target_dir=test_dir,
                                      transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, train_dataset.classes


def mfcc_transform(y: np.ndarray,
                   sr: int,
                   mfcc_params: Dict
                   ) -> torch.Tensor:
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=mfcc_params['n_fft'],
        win_length=mfcc_params['win_length'],
        hop_length=mfcc_params['hop_length'],
        n_mels=mfcc_params['n_mels'],
        htk=mfcc_params['htk'],
        norm=mfcc_params['norm_melspec'],
    )
    mfcc_librosa = librosa.feature.mfcc(
        S=librosa.core.spectrum.power_to_db(melspec),
        n_mfcc=mfcc_params['n_mfcc'],
        dct_type=mfcc_params['dct_type'],
        norm=mfcc_params['norm_mfcc']
    )

    return torch.from_numpy(mfcc_librosa)

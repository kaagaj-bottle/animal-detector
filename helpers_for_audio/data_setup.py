import torch
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Tuple, List, Dict

from torchvision import datasets, transforms
import librosa
import soundfile


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:

    classes = sorted(item.name for item in os.scandir(
        directory) if item.is_dir())

    class_to_idx = {classname: i for i, classname in enumerate(classes)}

    return classes, class_to_idx


class CustomAudioDataset(Dataset):

    def __init__(self, target_dir: str, transform=None, sr: int = 16000) -> None:
        self.paths = list(Path(target_dir).glob("*/*.wav"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)
        self.sr = sr

    def load_audio(self, index: int) -> torch.Tensor:
        audio_path = self.paths[index]
        audio, _ = librosa.load(audio_path, sr=self.sr)
        audio = torch.from_numpy(audio)
        return audio
    def __len__(self)->int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        audio = self.load_audio(index)
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

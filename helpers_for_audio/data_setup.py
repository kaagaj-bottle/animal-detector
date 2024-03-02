import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict
import librosa


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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        audio = self.load_audio(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(audio), class_idx
        else:
            return audio, class_idx

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class LabelsFile:
    path: Path
    sep: str = " "

class AudioDataset(Dataset):
    def __init__(self, data_dir: Path, nb_frames: int, labels: LabelsFile):
        self.data_dir = Path(data_dir)
        df = pd.read_csv(labels.path, engine="python", sep=labels.sep)
        self.vid = df["Vid_name"].tolist()
        self.y = (df["Label"] - 1).astype(int).tolist()
        self.nb_frames = nb_frames

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        spec = torch.load(self.data_dir / f"{self.vid[idx]}.mp4.spec.{self.nb_frames}.pt")
        return spec, self.y[idx]

class VGAFDataset(Dataset):
    def __init__(self, images_dir: Path, audio_dir: Path, nb_frames_img: int, nb_frames_audio: int, labels: LabelsFile):
        df = pd.read_csv(labels.path, engine="python", sep=labels.sep)
        self.vid = df["Vid_name"].tolist()
        self.y = (df["Label"] - 1).astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.audio_dir = Path(audio_dir)
        self.nb_frames_img = nb_frames_img
        self.nb_frames_audio = nb_frames_audio

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        img = torch.load(self.images_dir / f"{self.vid[idx]}.mp4.img.{self.nb_frames_img}.pt")
        spec = torch.load(self.audio_dir / f"{self.vid[idx]}.mp4.spec.{self.nb_frames_audio}.pt")
        return img, spec, self.y[idx]

class SyntheticDataset(Dataset):
    def __init__(self, images_root: Path, audio_root: Path, nb_frames_img: int, nb_frames_audio: int, class_name: str, labels_path: Path):
        df = pd.read_csv(labels_path, engine="python", sep=" ")
        self.vid = df["Vid_name"].tolist()
        self.y = df["Label"].astype(int).tolist()

        self.images_dir = Path(images_root) / class_name
        self.audio_dir = Path(audio_root) / class_name
        self.nb_frames_img = nb_frames_img
        self.nb_frames_audio = nb_frames_audio

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        img = torch.load(self.images_dir / f"{self.vid[idx]}.synt.img.{self.nb_frames_img}.pt")
        spec = torch.load(self.audio_dir / f"{self.vid[idx]}.synt.spec.{self.nb_frames_audio}.pt")
        return img, spec, self.y[idx]

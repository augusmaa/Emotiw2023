from __future__ import annotations
import random
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, Subset
from .datasets import VGAFDataset, SyntheticDataset, LabelsFile

def build_train_dataset(
    vgaf_images_dir: Path,
    vgaf_audio_dir: Path,
    labels_train: Path,
    synt_images_root: Path,
    synt_audio_root: Path,
    nb_frames_img: int,
    nb_frames_audio: int,
    synt_rate: float,
    seed: int = 2023,
):
    labels = LabelsFile(labels_train)
    ds_vgaf = VGAFDataset(vgaf_images_dir, vgaf_audio_dir, nb_frames_img, nb_frames_audio, labels)

    if synt_rate <= 0.0:
        return ds_vgaf

    ds1 = SyntheticDataset(synt_images_root, synt_audio_root, nb_frames_img, nb_frames_audio, "Positive", synt_images_root / "Positive_labels.txt")
    ds2 = SyntheticDataset(synt_images_root, synt_audio_root, nb_frames_img, nb_frames_audio, "Neutral",  synt_images_root / "Neutral_labels.txt")
    ds3 = SyntheticDataset(synt_images_root, synt_audio_root, nb_frames_img, nb_frames_audio, "Negative", synt_images_root / "Negative_labels.txt")
    ds_synt = ConcatDataset([ds1, ds2, ds3])

    if synt_rate >= 1.0:
        return ds_synt

    # sample synt subset to match your original proportion rule
    random.seed(seed)
    total_synt = len(ds_synt)
    subset_size = round(((len(ds_vgaf) * synt_rate) / (1 - synt_rate)) / 3) * 3
    subset_idx = random.sample(range(total_synt), subset_size)
    ds_synt_subset = Subset(ds_synt, subset_idx)

    return ConcatDataset([ds_vgaf, ds_synt_subset])

def build_val_dataset(vgaf_images_dir: Path, vgaf_audio_dir: Path, labels_val: Path, nb_frames_img: int, nb_frames_audio: int):
    return VGAFDataset(vgaf_images_dir, vgaf_audio_dir, nb_frames_img, nb_frames_audio, LabelsFile(labels_val))

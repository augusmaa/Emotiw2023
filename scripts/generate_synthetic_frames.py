#!/usr/bin/env python3
"""
Generate synthetic "video" frame sequences by pasting face cutouts (with alpha)
onto random background images.

Output structure:
  <out_dir>/<label>/<video_name>/<frame_idx>.png
  <out_dir>/<label>_labels.txt   (Vid_name Label)

Example:
  python scripts/generate_synthetic_frames.py \
    --bg_root /data/LSUN_dataset \
    --faces_root /data/Data_faces/All_faces_train_rmbg \
    --label Positive \
    --videos_per_bg 200 \
    --frames_per_video 75 \
    --out_dir data/synthetic_videos \
    --overwrite
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


LABEL_TO_ID: Dict[str, int] = {"Positive": 0, "Neutral": 1, "Negative": 2}


@dataclass(frozen=True)
class GenConfig:
    bg_root: Path
    faces_root: Path
    out_dir: Path
    label: str

    videos_per_bg: int = 200
    frames_per_video: int = 75

    canvas_size: int = 326  # final background resized to (canvas_size, canvas_size)
    base_face_size: int = 350  # before scaling
    face_scale_min: float = 0.10
    face_scale_max: float = 0.15
    rotate_min_deg: int = -45
    rotate_max_deg: int = 45
    rotate_prob: float = 0.5

    min_faces_per_video: int = 3
    max_faces_per_video: int = 15

    margin: int = 30
    drift_per_frame: int = 1  # increases minimum margin as frame index increases (like your original code)
    max_overlap_ratio: float = 0.01  # overlap_area / new_face_area
    max_tries_per_face: int = 80

    seed: int = 2023
    overwrite: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def list_subdirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]


def load_bg(path: Path, canvas_size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((canvas_size, canvas_size), resample=Image.BILINEAR)


def load_face_rgba(path: Path, base_size: int) -> Image.Image:
    # Face cutouts should have alpha (RGBA). If not, we still convert, but overlap logic relies on alpha.
    img = Image.open(path).convert("RGBA")
    return img.resize((base_size, base_size), resample=Image.BILINEAR)


def random_transform_face(face: Image.Image, cfg: GenConfig) -> Image.Image:
    # Scale
    scale = random.uniform(cfg.face_scale_min, cfg.face_scale_max)
    w, h = face.size
    face = face.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=Image.BILINEAR)

    # Rotation
    if random.random() < cfg.rotate_prob:
        deg = random.randint(cfg.rotate_min_deg, cfg.rotate_max_deg)
        face = face.rotate(deg, resample=Image.BILINEAR, expand=True)

    return face


def alpha_mask(face_rgba: Image.Image, alpha_threshold: int = 10) -> np.ndarray:
    """
    Returns a boolean mask where the face is "present" based on alpha channel.
    """
    a = np.array(face_rgba.split()[-1])  # alpha channel
    return a > alpha_threshold


def sample_position(
    canvas_size: int, face_w: int, face_h: int, margin: int
) -> Tuple[int, int]:
    x_max = canvas_size - margin - face_w
    y_max = canvas_size - margin - face_h
    if x_max <= margin or y_max <= margin:
        # face too large; clamp to origin-ish
        return margin, margin
    x = random.randint(margin, x_max)
    y = random.randint(margin, y_max)
    return x, y


def place_face_non_overlapping(
    bg: Image.Image,
    occ: np.ndarray,
    face_rgba: Image.Image,
    cfg: GenConfig,
    frame_idx: int,
) -> Tuple[Image.Image, np.ndarray, bool]:
    """
    Try to place face_rgba on bg with overlap constraint using occ mask.
    Returns: (updated_bg, updated_occ, placed_ok)
    """
    mask = alpha_mask(face_rgba)
    area = int(mask.sum())
    if area == 0:
        return bg, occ, False

    face_w, face_h = face_rgba.size
    margin = cfg.margin + frame_idx * cfg.drift_per_frame

    for _ in range(cfg.max_tries_per_face):
        x, y = sample_position(cfg.canvas_size, face_w, face_h, margin)

        # bounds check
        if x < 0 or y < 0 or x + face_w > cfg.canvas_size or y + face_h > cfg.canvas_size:
            continue

        region_occ = occ[y : y + face_h, x : x + face_w]
        overlap_area = int((region_occ & mask).sum())
        overlap_ratio = overlap_area / max(area, 1)

        if overlap_ratio <= cfg.max_overlap_ratio:
            # paste using alpha
            bg.paste(face_rgba.convert("RGB"), (x, y), face_rgba)
            region_occ |= mask
            occ[y : y + face_h, x : x + face_w] = region_occ
            return bg, occ, True

    return bg, occ, False


def ensure_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            # remove directory tree
            for p in sorted(path.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            path.rmdir()
        else:
            raise FileExistsError(f"Output path exists: {path} (use --overwrite to replace)")
    path.mkdir(parents=True, exist_ok=True)


def generate_for_label(cfg: GenConfig) -> None:
    if cfg.label not in LABEL_TO_ID:
        raise ValueError(f"label must be one of {list(LABEL_TO_ID.keys())}, got {cfg.label}")

    label_id = LABEL_TO_ID[cfg.label]
    label_out = cfg.out_dir / cfg.label

    ensure_dir(label_out, overwrite=cfg.overwrite)

    face_dir = cfg.faces_root / cfg.label
    if not face_dir.exists():
        raise FileNotFoundError(f"Face directory not found: {face_dir}")

    bg_categories = list_subdirs(cfg.bg_root)
    if not bg_categories:
        raise FileNotFoundError(f"No background subfolders found in: {cfg.bg_root}")

    all_label_entries: List[Tuple[str, int]] = []

    for cat in bg_categories:
        bg_images = list_images(cat)
        if not bg_images:
            continue

        sample_bg = random.sample(bg_images, k=min(cfg.videos_per_bg, len(bg_images)))

        for vid_idx, bg_path in enumerate(sample_bg):
            video_name = f"{cat.name}_Video_{vid_idx}"
            video_dir = label_out / video_name
            video_dir.mkdir(parents=True, exist_ok=True)

            # choose faces once per video (same set reused across all frames like your original logic)
            n_faces = random.randint(cfg.min_faces_per_video, cfg.max_faces_per_video)
            face_files = list_images(face_dir)
            if len(face_files) == 0:
                raise FileNotFoundError(f"No face images found in: {face_dir}")

            chosen_faces = random.sample(face_files, k=min(n_faces, len(face_files)))

            for fidx in range(cfg.frames_per_video):
                bg = load_bg(bg_path, cfg.canvas_size)
                occ = np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=bool)

                for face_path in chosen_faces:
                    face = load_face_rgba(face_path, cfg.base_face_size)
                    face = random_transform_face(face, cfg)

                    bg, occ, _ = place_face_non_overlapping(bg, occ, face, cfg, frame_idx=fidx)

                out_path = video_dir / f"{fidx}.png"
                bg.save(out_path)

            all_label_entries.append((video_name, label_id))
            print(f"[{cfg.label}] generated {video_name} ({cfg.frames_per_video} frames)")

    # write labels file (Vid_name Label)
    labels_path = cfg.out_dir / f"{cfg.label}_labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        f.write("Vid_name Label\n")
        for vid_name, lab in all_label_entries:
            f.write(f"{vid_name} {lab}\n")

    print(f"Done. Wrote labels: {labels_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic video frames for EmotiW 2023.")
    p.add_argument("--bg_root", type=Path, required=True, help="Root directory containing background subfolders (e.g., LSUN categories).")
    p.add_argument("--faces_root", type=Path, required=True, help="Root directory containing faces/<label>/ images with alpha.")
    p.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--label", type=str, required=True, choices=list(LABEL_TO_ID.keys()))

    p.add_argument("--videos_per_bg", type=int, default=200)
    p.add_argument("--frames_per_video", type=int, default=75)

    p.add_argument("--canvas_size", type=int, default=326)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--overwrite", action="store_true", help="Overwrite output label directory if it exists.")

    # face placement/appearance
    p.add_argument("--min_faces_per_video", type=int, default=3)
    p.add_argument("--max_faces_per_video", type=int, default=15)
    p.add_argument("--face_scale_min", type=float, default=0.10)
    p.add_argument("--face_scale_max", type=float, default=0.15)
    p.add_argument("--rotate_prob", type=float, default=0.5)
    p.add_argument("--rotate_min_deg", type=int, default=-45)
    p.add_argument("--rotate_max_deg", type=int, default=45)
    p.add_argument("--margin", type=int, default=30)
    p.add_argument("--drift_per_frame", type=int, default=1)
    p.add_argument("--max_overlap_ratio", type=float, default=0.001)
    p.add_argument("--max_tries_per_face", type=int, default=80)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenConfig(
        bg_root=args.bg_root,
        faces_root=args.faces_root,
        out_dir=args.out_dir,
        label=args.label,
        videos_per_bg=args.videos_per_bg,
        frames_per_video=args.frames_per_video,
        canvas_size=args.canvas_size,
        seed=args.seed,
        overwrite=args.overwrite,
        min_faces_per_video=args.min_faces_per_video,
        max_faces_per_video=args.max_faces_per_video,
        face_scale_min=args.face_scale_min,
        face_scale_max=args.face_scale_max,
        rotate_prob=args.rotate_prob,
        rotate_min_deg=args.rotate_min_deg,
        rotate_max_deg=args.rotate_max_deg,
        margin=args.margin,
        drift_per_frame=args.drift_per_frame,
        max_overlap_ratio=args.max_overlap_ratio,
        max_tries_per_face=args.max_tries_per_face,
    )

    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    generate_for_label(cfg)


if __name__ == "__main__":
    main()

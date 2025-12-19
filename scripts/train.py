from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from emotiw2023.utils.seed import set_seed
from emotiw2023.utils.checkpoint import save_checkpoint, load_checkpoint
from emotiw2023.utils.notify import Notifier
from emotiw2023.losses import FocalLoss
from emotiw2023.data.datasets import AudioDataset, LabelsFile
from emotiw2023.data.builders import build_train_dataset, build_val_dataset
from emotiw2023.models.audio import AudioClassifier, AudioClassifierFlatten
from emotiw2023.models.video import VideoClassifier, VideoClassifierFlatten
from emotiw2023.models.fusion import VideoAudioFusion, VideoAudioFusionFlatten
from emotiw2023.training.engine import train_one_epoch, validate

def parse_args():
    p = argparse.ArgumentParser("EmotiW 2023 training (clean repo version)")

    p.add_argument("--train_corpus", choices=["audio", "video", "fusion"], default="fusion")
    p.add_argument("--nb_frames_audio", type=int, default=5)
    p.add_argument("--nb_frames_video", type=int, default=5)
    p.add_argument("--flatten", action="store_true")

    p.add_argument("--hidden_ff_audio", type=int, default=2048)
    p.add_argument("--num_heads_audio", type=int, default=1)
    p.add_argument("--num_heads_crossatt", type=int, default=1)
    p.add_argument("--dropout_ff", type=float, default=0.5)
    p.add_argument("--dropout_att", type=float, default=0.2)

    p.add_argument("--dir_data_parent", type=Path, default=Path("/Corpora/VGAF/"))
    p.add_argument("--synt_rate", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resume_from", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("./runs"))

    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--notify", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed, deterministic=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    notifier = Notifier() if args.notify else None

    labels_train = args.dir_data_parent / "Train_labels.txt"
    labels_val = args.dir_data_parent / "Val_labels.txt"

    # Resolve data folders 
    if args.nb_frames_audio == 5 and args.nb_frames_video == 5:
        vgaf_train = args.dir_data_parent / "VGAF_Tensors_all_5/Train"
        vgaf_val = args.dir_data_parent / "VGAF_Tensors_all_5/Val"
        synt_root = args.dir_data_parent / "Synthetic2661_all_5/"
    elif args.nb_frames_audio == 5 and args.nb_frames_video == 75:
        vgaf_train = args.dir_data_parent / "VGAF_Tensors_all_75/Train"
        vgaf_val = args.dir_data_parent / "VGAF_Tensors_all_75/Val"
        synt_root = args.dir_data_parent / "Synthetic2661_all_75/"
    else:
        vgaf_train = args.dir_data_parent / "VGAF_Tensors_all_75/Train"
        vgaf_val = args.dir_data_parent / "VGAF_Tensors_all_75/Val"
        synt_root = args.dir_data_parent / "Synthetic2661_all_75/"

    # Build model
    if args.train_corpus == "audio":
        model = (AudioClassifierFlatten if args.flatten else AudioClassifier)(
            seq_len=args.nb_frames_audio,
            d_hidden_ff=args.hidden_ff_audio,
            num_heads=args.num_heads_audio,
            dropout_ff=args.dropout_ff,
            dropout_att=args.dropout_att,
        )
        train_ds = AudioDataset(vgaf_train, args.nb_frames_audio, LabelsFile(labels_train))
        val_ds = AudioDataset(vgaf_val, args.nb_frames_audio, LabelsFile(labels_val))
        mode = "audio"

    elif args.train_corpus == "video":
        model = (VideoClassifierFlatten if args.flatten else VideoClassifier)()
        train_ds = build_train_dataset(vgaf_train, vgaf_train, labels_train, synt_root, synt_root,
                                       args.nb_frames_video, args.nb_frames_audio, args.synt_rate, seed=args.seed)
        val_ds = build_val_dataset(vgaf_val, vgaf_val, labels_val, args.nb_frames_video, args.nb_frames_audio)
        mode = "video"

    else:
        model = (VideoAudioFusionFlatten if args.flatten else VideoAudioFusion)(
            seq_len_audio=args.nb_frames_audio,
            seq_len_video=args.nb_frames_video,
            d_hidden_ff_audio=args.hidden_ff_audio,
            num_heads_audio=args.num_heads_audio,
            dropout_ff=args.dropout_ff,
            dropout_att=args.dropout_att,
            num_heads_cross=args.num_heads_crossatt,
        )
        train_ds = build_train_dataset(vgaf_train, vgaf_train, labels_train, synt_root, synt_root, # we use vgaf_train twice because the same director path is for audio and video, the difference is the extention
                                       args.nb_frames_video, args.nb_frames_audio, args.synt_rate, seed=args.seed)
        val_ds = build_val_dataset(vgaf_val, vgaf_val, labels_val, args.nb_frames_video, args.nb_frames_audio)
        mode = "fusion"

    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = FocalLoss(args.alpha, args.gamma) if args.use_focal else torch.nn.CrossEntropyLoss()

    run_name = f"{args.train_corpus}_a{args.nb_frames_audio}_v{args.nb_frames_video}_flat{args.flatten}_syn{int(args.synt_rate*100)}"
    tb = SummaryWriter(log_dir=str(args.out_dir / "tb" / run_name))
    ckpt_dir = args.out_dir / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    start_epoch = 0
    if args.resume_from is not None and args.resume_from.exists():
        ck = load_checkpoint(args.resume_from, model, optimizer, map_location=device)
        start_epoch = ck.epoch + 1
        best_acc = ck.best_metric

    for epoch in range(start_epoch, args.num_epochs):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, mode)
        va = validate(model, val_loader, criterion, device, mode)

        tb.add_scalars("Loss", {"train": tr.loss, "val": va.loss}, epoch)
        tb.add_scalars("Accuracy", {"train": tr.acc, "val": va.acc}, epoch)

        if va.acc > best_acc:
            best_acc = va.acc
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best_acc)
            if notifier:
                notifier.send(f"[{run_name}] New best val acc={best_acc:.4f} @ epoch {epoch}")

        print(f"Epoch {epoch:03d} | train: loss={tr.loss:.4f} acc={tr.acc:.4f} | val: loss={va.loss:.4f} acc={va.acc:.4f}")

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, args.num_epochs - 1, best_acc)
    if notifier:
        notifier.send(f"[{run_name}] Training finished. Best val acc={best_acc:.4f}")

if __name__ == "__main__":
    main()

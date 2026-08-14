import argparse
from pathlib import Path
from typing import cast
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CXP_dataset
from model import _build_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["medsiglip", "medimageinsight"],
        help="Feature extractor backbone.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory above /CheXpert-v1.0-small",
    )
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=Path("csv_data"),
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--features_dir",
        type=Path,
        default=Path("features_cache"),
        help="Output directory for cached features",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Extract features only for a small subset of data.",
    )
    args = parser.parse_args()

    args.features_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    print(f"Loading encoder for {args.backbone}...")
    encoder, _ = _build_encoder(args.backbone)
    encoder = encoder.to(device)
    encoder.eval()

    splits = [
        ("train", "train_drain_shortcut.csv"),
        ("train_v2", "train_drain_shortcut_v2.csv"),
        ("val", "val_drain_shortcut.csv"),
        ("val_v2", "val_drain_shortcut_v2.csv"),
        ("test_aligned", "test_drain_shortcut_aligned.csv"),
        ("test_misaligned", "test_drain_shortcut_misaligned.csv"),
    ]

    for split_name, csv_name in splits:
        csv_path = args.csv_dir / csv_name
        if not csv_path.exists():
            print(f"Skipping {csv_name} (not found)")
            continue

        out_file = args.features_dir / f"{args.backbone}_{split_name}.pt"
        if out_file.exists():
            print(f"Skipping split {split_name}: {out_file} already exists.")
            continue

        print(f"Processing split: {split_name} ({csv_name})")

        dataset = CXP_dataset(
            root_dir=args.data_dir,
            csv_file=csv_path,
            augment=False,
            compute_sample_weights=False,
            split_name=split_name,
            backbone=args.backbone,
            use_cached_features=False,
        )

        if args.debug:
            dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), 50)))

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        features_dict = {}
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting {split_name}"):
                batch_data = batch
                imgs = batch_data.inputs.to(device)

                features = encoder(imgs)
                features = features.cpu()  # Keep features on CPU to save memory

                # dataset could be a Subset in debug mode
                base_dataset = (
                    dataset.dataset
                    if isinstance(dataset, torch.utils.data.Subset)
                    else dataset
                )
                base_dataset = cast(CXP_dataset, base_dataset)

                for idx_in_batch, original_idx in enumerate(batch_data.indices):
                    original_idx = original_idx.item()
                    path_str = str(base_dataset.path.iloc[original_idx])
                    features_dict[path_str] = features[idx_in_batch].clone()

        torch.save(features_dict, out_file)
        print(f"Saved {len(features_dict)} features to {out_file}\n")

    print("Feature extraction complete!")


if __name__ == "__main__":
    main()

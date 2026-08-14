import os
import logging
import torch
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms.v2 as transforms
from pathlib import Path
from PIL import Image
from typing import NamedTuple, Optional


class BatchData(NamedTuple):
    indices: torch.Tensor
    inputs: torch.Tensor
    labels: torch.Tensor
    drains: torch.Tensor
    weights: torch.Tensor

    @property
    def usable_weights(self) -> Optional[torch.Tensor]:
        if (
            torch.is_floating_point(self.weights)
            and not torch.isfinite(self.weights).any()
        ):
            return None
        return self.weights

    def extra_info(self) -> dict[str, torch.Tensor]:
        return {"drain": self.drains, "indices": self.indices}


def grayscale_to_rgb(i):
    return torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i


class HFImageTransform:
    """
    Preprocessing pipeline for HuggingFace models (MedSigLIP, MedImageInsight).
    Augmentation (PIL-space) runs before the model's AutoProcessor handles
    resize / rescale / normalize.
    """

    def __init__(self, model_id_or_path: str, augment: bool):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self._to_pil = transforms.ToPILImage()
        self.aug = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(degrees=7),
                ]
            )
            if augment
            else None
        )

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = self._to_pil(img)
        img = img.convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


class MedImageInsightTransform:
    def __init__(self, repo_root: str, augment: bool):
        import sys
        from pathlib import Path

        repo_root = Path(repo_root).resolve()
        sys.path.insert(0, str(repo_root))

        from MedImageInsight.Utils.Arguments import load_opt_from_config_files
        from MedImageInsight.ImageDataLoader import build_transforms

        model_dir = repo_root / "2024.09.27"
        opt = load_opt_from_config_files([str(model_dir / "config.yaml")])

        self.preprocess = build_transforms(opt, False)
        self._to_pil = transforms.ToPILImage()
        self.aug = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(degrees=7),
                ]
            )
            if augment
            else None
        )

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = self._to_pil(img)
        img = img.convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
        return self.preprocess(img)


def build_transform(backbone: str, augment: bool):
    """Return the appropriate transform pipeline for the given backbone."""
    if backbone != "densenet":
        augment = False

    if backbone == "densenet":
        if augment:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Lambda(grayscale_to_rgb),
                    transforms.Normalize(  # params for pretrained resnet, see https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.DenseNet121_Weights
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomRotation(degrees=20),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomResizedCrop(
                        size=224, scale=(0.7, 1.0), ratio=(0.75, 1.3)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Lambda(grayscale_to_rgb),
                    transforms.Normalize(  # params for pretrained resnet, see https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.DenseNet121_Weights
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        return transform

    elif backbone == "medsiglip":
        return HFImageTransform("google/medsiglip-448", augment=augment)

    elif backbone == "medimageinsight":
        return MedImageInsightTransform(
            os.environ["MEDIMAGEINSIGHT_REPO"],
            augment=augment,
        )

    else:
        raise ValueError(
            f"Unknown backbone: '{backbone}'. "
            f"Choose from: 'densenet', 'medsiglip', 'medimageinsight'."
        )


class CXP_dataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root_dir: Path | str,
        csv_file: Path | str,
        augment: bool = True,
        compute_sample_weights: bool = False,
        split_name: str | None = None,
        backbone: str = "densenet",
        use_cached_features: bool = False,
        features_dir: Path | str | None = None,
    ) -> None:

        self.use_cached_features = use_cached_features and backbone != "densenet"
        if self.use_cached_features:
            transform = None
        else:
            transform = build_transform(backbone, augment)

        super().__init__(root_dir, transform)

        self.split_name = split_name if split_name is not None else Path(csv_file).stem
        self.csv_file = Path(csv_file)

        # Enforce pure alignment/misalignment for test sets by redistributing the 20% noise
        if self.split_name in ["test_aligned", "test_misaligned"]:
            base_dir = Path(csv_file).parent
            aligned_csv = base_dir / "test_drain_shortcut_aligned.csv"
            misaligned_csv = base_dir / "test_drain_shortcut_misaligned.csv"

            if aligned_csv.exists() and misaligned_csv.exists():
                df_aligned = pd.read_csv(aligned_csv)
                df_misaligned = pd.read_csv(misaligned_csv)
                total_test = pd.concat([df_aligned, df_misaligned], ignore_index=True)
                total_test = total_test.drop_duplicates(subset=["Path"])

                drain_col = pd.to_numeric(total_test.Drain, errors="coerce")
                if self.split_name == "test_aligned":
                    df = total_test[total_test.Pneumothorax == drain_col]
                else:
                    df = total_test[
                        (total_test.Pneumothorax != drain_col) & ~drain_col.isna()
                    ]

                assert len(df) == 50
            else:
                raise RuntimeError(
                    f"Expected aligned and misaligned CSVs not found for test split: {aligned_csv}, {misaligned_csv}. "
                )
        else:
            df = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.path = df.Path.str.replace(
            "CheXpert-v1.0/", "CheXpert-v1.0-small/", regex=False
        )
        self.idx = df.index
        self.transform = transform

        if self.use_cached_features:
            if features_dir is None:
                raise ValueError(
                    "features_dir must be provided when use_cached_features is True"
                )
            
            # Map custom split names back to the original cached feature files
            if self.split_name in ["test_aligned", "test_misaligned"]:
                cache_split = self.split_name
            else:
                csv_stem = Path(csv_file).stem
                if csv_stem == "train_drain_shortcut":
                    cache_split = "train"
                elif csv_stem == "train_drain_shortcut_v2":
                    cache_split = "train_v2"
                elif csv_stem == "val_drain_shortcut":
                    cache_split = "val"
                elif csv_stem == "val_drain_shortcut_v2":
                    cache_split = "val_v2"
                else:
                    cache_split = self.split_name

            feature_file = Path(features_dir) / f"{backbone}_{cache_split}.pt"
            if not feature_file.exists():
                raise FileNotFoundError(
                    f"Cached features not found: {feature_file}. "
                    f"Please run extract_features.py --backbone {backbone} first."
                )
            self.features_dict = torch.load(feature_file, weights_only=True)
            logging.info(
                f"[split={self.split_name}] Loaded {len(self.features_dict)} cached features from {feature_file}"
            )
        else:
            self.features_dict = None

        self.labels = df.Pneumothorax.astype(int)
        # Drain stays float because some rows legitimately carry NaN for missing group labels.
        self.drain = pd.to_numeric(df.Drain, errors="coerce").astype(np.float32)

        # Drop rows whose image path is missing to avoid repeated runtime retries/log spam.
        exists_mask = self.path.map(
            lambda p: os.path.exists(os.path.join(self.root_dir, str(p)))
        ).to_numpy()
        if not exists_mask.all():
            missing_count = int((~exists_mask).sum())
            logging.warning(
                "[split=%s] Dropping %d/%d rows with missing image files from %s",
                self.split_name,
                missing_count,
                len(exists_mask),
                self.csv_file,
            )
            self.path = self.path[exists_mask].reset_index(drop=True)
            self.idx = self.idx[exists_mask]
            self.labels = self.labels[exists_mask].reset_index(drop=True)
            self.drain = self.drain[exists_mask].reset_index(drop=True)

        n_total = int(len(self.labels))
        n_drain_known = int((~self.drain.isna()).sum())
        n_drain_missing = int(self.drain.isna().sum())
        logging.info(
            "[split=%s] Loaded dataset rows=%d from %s (drain_known=%d, drain_missing=%d, compute_sample_weights=%s)",
            self.split_name,
            n_total,
            self.csv_file,
            n_drain_known,
            n_drain_missing,
            compute_sample_weights,
        )

        # We always return a weight tensor in BatchData. When compute_sample_weights=False,
        # the tensor stays NaN so downstream code treats the batch as unweighted.
        self.weights = np.full(len(self.labels), np.nan, dtype=np.float32)

        if compute_sample_weights:
            # Compute sample weights
            # We use effective number of samples balancing as per
            # https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
            valid_mask = ~self.drain.isna().to_numpy()

            if valid_mask.any():
                groups = (
                    (self.labels[valid_mask] * 2 + self.drain[valid_mask])
                    .astype(int)
                    .to_numpy()
                )
                counts = np.bincount(groups, minlength=4)
                assert np.all(counts > 0), (
                    f"Expected all 4 groups to be present at least once, got counts={counts.tolist()}"
                )
                beta = 0.9999
                effective_num = (1 - beta**counts) / (1 - beta)
                class_weights = np.divide(
                    1.0,
                    effective_num,
                    out=np.zeros_like(effective_num, dtype=np.float64),
                    where=counts > 0,
                )
                normalization_factor = valid_mask.sum() / class_weights[groups].sum()
                normalized_class_weights = class_weights * normalization_factor
                self.weights[valid_mask] = normalized_class_weights[groups]
                assert np.isclose(
                    np.nansum(self.weights), valid_mask.sum(), rtol=1e-4
                )  # relax check for float32

                logging.info(
                    "[split=%s] Sample-weight groups @ beta=0.9999: y0_g0 n=%d w=%.7f, y0_g1 n=%d w=%.7f, "
                    "y1_g0 n=%d w=%.7f, y1_g1 n=%d w=%.7f",
                    self.split_name,
                    int(counts[0]),
                    float(normalized_class_weights[0]),
                    int(counts[1]),
                    float(normalized_class_weights[1]),
                    int(counts[2]),
                    float(normalized_class_weights[2]),
                    int(counts[3]),
                    float(normalized_class_weights[3]),
                )

            n_invalid = int((~valid_mask).sum())
            n_valid = int(valid_mask.sum())
            n_total = int(len(valid_mask))
            if n_invalid > 0:
                logging.warning(
                    "[split=%s] compute_sample_weights=True: Drain is NaN for %d/%d rows (valid weights for %d/%d). "
                    "Those rows keep NaN sample weights and are excluded from weighted terms.",
                    self.split_name,
                    n_invalid,
                    n_total,
                    n_valid,
                    n_total,
                )

    def __getitem__(self, index: int) -> BatchData:
        path_str = str(self.path.iloc[index])

        if self.use_cached_features and self.features_dict is not None:
            if path_str in self.features_dict:
                img = self.features_dict[path_str].float()
            else:
                raise RuntimeError(
                    f"Feature not found for path: {path_str}. Please ensure features are properly extracted."
                )
        else:
            full_path = os.path.join(self.root_dir, path_str)
            transform = self.transform
            assert transform is not None

            try:
                img = torchvision.io.read_image(full_path)
                img = transform(img)
            except (RuntimeError, FileNotFoundError, OSError, ValueError) as e:
                if os.path.exists(full_path):
                    try:
                        img = transform(Image.open(full_path).convert("RGB"))
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"Fallback load failed at index {index} for {full_path}. "
                            f"Original error: {e}. Fallback error: {fallback_error}"
                        )
                else:
                    raise FileNotFoundError(
                        f"Image not found at index {index}: {full_path}. "
                        f"Original error: {e}"
                    )

        return BatchData(
            indices=torch.tensor(index, dtype=torch.long),
            inputs=img,
            labels=torch.tensor(int(self.labels.iloc[index]), dtype=torch.long),
            drains=torch.tensor(float(self.drain.iloc[index]), dtype=torch.float32),
            weights=torch.tensor(float(self.weights[index]), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.path)

"""
Dataset + DataLoader for Webots experience frames -> (image, [x, y]).

"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, decode_image

from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from torch.utils.data import Subset
import time
from typing import Optional



# --- little helper functions for timing and reproducibility ---
@contextmanager
def timer(name: str) -> None:
    """
    Context manager to measure execution time of code blocks.
    """
    start_time = time.time()
    yield
    elapsed_seconds = time.time() - start_time

    # Format time as hh:mm:ss
    hours, remainder = divmod(int(elapsed_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((elapsed_seconds - int(elapsed_seconds)) * 1000)

    if hours > 0:
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    print(f"{name} completed in {time_str}")


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    else:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_parameters(model):
    """Get number of parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# --- subset sampling ----------------------------------------------------------------------
class SubsetSampler(ABC):
    """
    Subclasses rank all samples; a subset of size n is the top n.
    The ranking must not depend on n, so subsets of increasing size are nested.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def rank(self, df: pd.DataFrame) -> np.ndarray:
        """Permutation of positional indices into df, highest priority first."""

    def select(
        self, 
        df: pd.DataFrame, 
        size: int | float, 
        pool: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """size: int = count, float = fraction of pool. Returns positional indices into df."""
        pool = np.arange(len(df)) if pool is None else np.asarray(pool)
        n = round(size * len(pool)) if isinstance(size, float) else size
        return pool[self.rank(df.iloc[pool])[:n]]


class StratifiedPoseSampler(SubsetSampler):
    """
    Round-robin over occupied (x, y, angle) bins: one random sample per bin, then a second, etc.
    Every prefix is as balanced across bins as possible; exhausted bins pass their budget on.
    Angle bins are circular and centred on 0°, so corridor-aligned headings don't straddle an edge.
    """

    def __init__(self, xy_bin_size: float, n_angle_bins: int = 8, seed: int = 0,
                 x_col: str = "X", y_col: str = "Y", z_col: str = "Z"):
        super().__init__(seed)
        self.xy_bin_size = xy_bin_size
        self.n_angle_bins = n_angle_bins
        self.cols = [x_col, y_col, z_col]

    def bins(self, df: pd.DataFrame) -> np.ndarray:
        x, y, z = (df[c].to_numpy(dtype=float) for c in self.cols)
        width = 360 / self.n_angle_bins
        keys = np.column_stack([
            np.floor((x - x.min()) / self.xy_bin_size),
            np.floor((y - y.min()) / self.xy_bin_size),
            np.floor(((z + width / 2) % 360) / width) % self.n_angle_bins,
        ])
        return np.unique(keys, axis=0, return_inverse=True)[1].ravel()

    def rank(self, df: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(df))
        b = self.bins(df)[perm]
        round_ = pd.Series(b).groupby(b).cumcount().to_numpy()  # k-th draw from its bin
        return perm[np.lexsort((rng.random(len(df)), round_))]  # by round, random bin order within


# --- dataset and dataloader for Webots frames ----------------------------------------------
class WebotsFrameDataset(Dataset):
    """
    One CSV row -> (frame_image, target_xy).

    :param data_csv:  Path to data.csv. Frames should be in the SAME directory as `<data_dir>/{Frame_ID:05d}.jpg`.
    :param transform: Compose applied to the image ONLY.
    :param id_col/x_col/y_col: column names, overridable if your column names differ.
    """

    def __init__(
        self,
        data_csv: str | Path,
        transform: Optional[Callable] = None,
        *,
        id_col: str = "Frame_ID",
        x_col: str = "X",
        y_col: str = "Y",
    ) -> None:

        self.data_csv = Path(data_csv)
        self.data_dir = self.data_csv.parent
        self.transform = transform
        self.id_col = id_col
        self.x_col = x_col
        self.y_col = y_col

        df = pd.read_csv(self.data_csv)
        missing = {id_col, x_col, y_col} - set(df.columns)
        if missing:
            raise ValueError(f"data.csv missing required columns: {sorted(missing)}")
        
        # Keep only what we use; positional index for __getitem__.
        self.df = df[[id_col, x_col, y_col]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    # frame path and loading
    def _frame_path(self, frame_id: int) -> Path:
        return self.data_dir / f"{int(frame_id):05d}.jpg"

    def _load_image(self, frame_id: int) -> torch.Tensor:
        return decode_image(self._frame_path(frame_id))

    def _load_coord(self, row: pd.Series) -> torch.Tensor:
        # add things here to get also the angle.
        return torch.tensor([row[self.x_col], row[self.y_col]], dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        row = self.df.iloc[idx]
        
        image = self._load_image(row[self.id_col])
        if self.transform is not None:
            image = self.transform(image)
        coord = self._load_coord(row)
        
        return image, coord


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32  # unique per worker, derived from base_seed
    import random, numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    # torch's per-worker seed is already set by PyTorch internals


def build_dataloader(
    data_csv: str | Path,
    transform: Optional[Callable] = None,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    **dataset_kwargs,
) -> DataLoader:

    dataset = WebotsFrameDataset(data_csv, transform=transform, **dataset_kwargs)
    if indices is not None:
        dataset = Subset(dataset, list(indices))
    
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if seed is not None else None,
        generator=generator,  # seeds the shuffle sampler
    )


if __name__ == "__main__":
    from torchvision.transforms import v2


    norm = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    csv = "../Denis/HIP_AE_VISUAL/Datasets/Tmaze_2/data.csv"

    # nested subsets for a data-efficiency sweep; pass pool=train_idx to sample within the train split
    sampler = StratifiedPoseSampler(xy_bin_size=0.25, n_angle_bins=8, seed=0)
    subsets = {size: sampler.select(pd.read_csv(csv), size) for size in [0.1, 0.25, 0.5, 1.0]}

    loader = build_dataloader(csv, transform=norm, batch_size=12, indices=subsets[0.5])
    images, xy = next(iter(loader))
    print(len(loader.dataset), len(loader.dataset.indices))
    print(images.shape, images.dtype)
    print(xy.shape, xy.dtype)

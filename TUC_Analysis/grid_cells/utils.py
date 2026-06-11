import math

import torch


def wavevectors(scale: float, orientation: float) -> torch.Tensor:
    """
    Three wavevectors at 60 deg separations.

    Magnitude: |k| = 4*pi / (sqrt(3) * scale).

    :return: wavevectors (3, 2).
    """
    kmag = 4.0 * torch.pi / (math.sqrt(3.0) * scale)
    offsets = torch.tensor([0.0, torch.pi / 3.0, 2.0 * torch.pi / 3.0])
    angles = orientation + offsets
    return kmag * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def jitter_phases(n_cells: int, jitter: float, gen) -> torch.Tensor:
    """
    Phase offsets on a jittered regular lattice over [0, 2pi)^2.

    :return: (n_cells, 2)
    """
    rows = int(torch.floor(torch.sqrt(torch.tensor(n_cells, dtype=torch.float32))))
    cols = math.ceil(n_cells / rows)

    ix, iy = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    cell_size = 2.0 * torch.pi / torch.tensor([rows, cols], dtype=torch.float32)

    grid = torch.stack([ix, iy], dim=-1).reshape(-1, 2).float()
    centers = grid * cell_size + cell_size / 2.0
    
    noise = (torch.rand(centers.shape, generator=gen) - 0.5) * cell_size * jitter
    phases = (centers + noise) % (2.0 * torch.pi)

    return phases[:n_cells]


def normalize_grid_rate(arg: torch.Tensor) -> torch.Tensor:
    """
    Shifted cosine sum, mapped to [0, 1].
    """
    return (2.0 / 3.0) * ((1.0 / 3.0) * torch.cos(arg).sum(dim=-1) + 0.5)
"""
Grid cell encoder.

Maps 2-D coordinates to grid-cell-like firing rates using the shifted
sum of three cosine gratings at 60 deg (Solstad, Moser & Einevoll 2006).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Optional

from grid_cells.utils import jitter_phases, normalize_grid_rate, wavevectors


@dataclass
class GridModule:
    scale: float
    orientation: float
    n_cells: int


class GridCellEncoder(nn.Module):
    def __init__(
        self,
        modules: list[GridModule],
        min_fr: float = 0.0,
        max_fr: float = 1.0,
        phase_jitter: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()

        self.min_fr = float(min_fr)
        self.max_fr = float(max_fr)
        self.module_specs = list(modules)

        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        k_list, phase_list = [], []
        for m in modules:
            k_m = wavevectors(m.scale, m.orientation)  # (3, 2)
            free_phases = jitter_phases(m.n_cells, phase_jitter, gen)  # (n, 2)
            # Project free phases (d1, d3) onto wavevectors.
            # The third phase d2 = d1 + d3 is implied by k2 = k1 + k3.
            phase_m = free_phases @ k_m.T  # (n, 3)
            k_list.append(k_m.unsqueeze(0).expand(m.n_cells, -1, -1))
            phase_list.append(phase_m)

        self.register_buffer("k", torch.cat(k_list, dim=0).float()) # (n, 3, 2)
        self.register_buffer("phase", torch.cat(phase_list, dim=0).float()) # (n, 3)

    @property
    def n_cells(self) -> int:
        return self.k.shape[0]

    @torch.no_grad()
    def forward(self, pos: torch.Tensor, noise_scale: float = 0.0) -> torch.Tensor:
        """
        :param pos: (B, 2) positions in metres
        :param noise_scale: std of IID Gaussian noise added to firing rates
        :return: (B, n) firing rates
        """
        # arg[b, n, c] = k[n, c] . pos[b] - phase[n, c]
        arg = torch.einsum("ncd,bd->bnc", self.k, pos.float()) - self.phase.unsqueeze(0)
        rate = self.min_fr + (self.max_fr - self.min_fr) * normalize_grid_rate(arg)

        if noise_scale > 0.0:
            rate = rate + noise_scale * torch.randn_like(rate)

        return rate


def _modules_to_dicts(modules: list[GridModule]) -> list[dict]:
    return [{"scale": m.scale, "orientation": m.orientation, "n_cells": m.n_cells}
            for m in modules]


def _dicts_to_modules(ds: list[dict]) -> list[GridModule]:
    return [GridModule(**d) for d in ds]


def save_grid_encoder(encoder: GridCellEncoder, path: str) -> None:
    torch.save({
        "config": {
            "modules": _modules_to_dicts(encoder.module_specs),
            "min_fr": encoder.min_fr,
            "max_fr": encoder.max_fr,
        },
        "state_dict": encoder.state_dict(),
    }, path)


def load_grid_encoder(path: str, map_location=None) -> GridCellEncoder:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = ckpt["config"]
    encoder = GridCellEncoder(
        modules=_dicts_to_modules(cfg["modules"]),
        min_fr=cfg["min_fr"],
        max_fr=cfg["max_fr"],
    )
    encoder.load_state_dict(ckpt["state_dict"])
    return encoder

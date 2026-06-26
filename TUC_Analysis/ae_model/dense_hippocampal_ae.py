import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Sequence


class MLP(nn.Module):
    """
    Generic MLP stack: input_dim -> hidden_dims -> output_dim.
    `output_activation` is an *instantiated* module (or None) appended after the
    final linear layer, so the same class serves both encoder (non-negative latent)
    and decoder (optionally bounded reconstruction).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation_cls=nn.ReLU,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        output_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PooledDenseAE(nn.Module):
    """
    Pool-then-MLP autoencoder over CNN feature maps.

    Pipeline:  feature map (B, C, H, W)
            - AdaptiveAvgPool2d(pool_output_size)        # the swappable "front end"
            - flatten                                    # obs_dim = C * oh * ow
            - [optional concat aux] -> MLP encoder -> h  # latent
            - MLP decoder -> reconstruct the POOLED vector (not the raw map)

    The reconstruction target is the pooled+flattened feature vector. Spatial
    structure finer than `pool_output_size` is discarded before the bottleneck.

    Swap points:
      - self.pool        : replace AdaptiveAvgPool2d with something else (e.g. max pool, attention, etc.)
      - self.encoder_mlp / self.decoder_mlp : any nn.Module with matching dims.
    """
    def __init__(
        self,
        n_hidden: int,
        in_channels: int = 512,
        pool_output_size: Sequence[int] = (1, 1),
        hidden_dims: Sequence[int] = (256, 128),
        activation_cls=nn.ReLU,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        latent_activation: Optional[type] = nn.ReLU,   # class or None; None -> linear bottleneck
        last_layer_activation: Optional[nn.Module] = None,  # instance or None; e.g. nn.Sigmoid()
        d_aux: Optional[int] = None,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.in_channels = in_channels
        self.pool_output_size = tuple(pool_output_size)
        self.d_aux = d_aux if d_aux is not None else 0
        self.last_layer_activation = last_layer_activation

        # swappable pooling, output is flattened anyway.
        self.pool = nn.AdaptiveAvgPool2d(self.pool_output_size)
        self.obs_dim = in_channels * self.pool_output_size[0] * self.pool_output_size[1]

        latent_act = latent_activation() if latent_activation is not None else None

        # Encoder consumes pooled features (+ aux), emits latent h
        self.encoder_mlp = MLP(
            input_dim=self.obs_dim + self.d_aux,
            hidden_dims=list(hidden_dims),
            output_dim=n_hidden,
            activation_cls=activation_cls,
            dropout=dropout,
            use_layernorm=use_layernorm,
            output_activation=latent_act,
        )

        # Decoder mirrors encoder (reversed widths), reconstructs the pooled vector
        self.decoder_mlp = MLP(
            input_dim=n_hidden,
            hidden_dims=list(reversed(list(hidden_dims))),
            output_dim=self.obs_dim + self.d_aux,
            activation_cls=activation_cls,
            dropout=dropout,
            use_layernorm=use_layernorm,
            output_activation=last_layer_activation,
        )

    # target / front-end helper, reused by training_step AND eval
    def pool_flatten(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x).flatten(1)            # B x obs_dim

    def encoder(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.pool_flatten(x)
        if aux is not None:
            z = torch.cat([z, aux], dim=1)
        return self.encoder_mlp(z)

    def decoder(self, h: torch.Tensor):
        # hmm.. do not like this, d_aux should also having a separate decoder with its own activation, but for now we just append it to the pooled vector.
        recon = self.decoder_mlp(h)
        x_recon = recon[:, :self.obs_dim]

        if self.d_aux > 0:
            aux_recon = recon[:, self.obs_dim:]
        else:
            aux_recon = None
        return x_recon, aux_recon

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None):
        h = self.encoder(x, aux)
        x_recon, aux_recon = self.decoder(h)
        return x_recon, aux_recon, h

    def training_step(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        x: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        C_factor: float = 1.0,
        alpha: float = 0,
        beta: float = 1.0,
    ):
        optimizer.zero_grad()
        x_recon, aux_recon, hidden = self.forward(x, aux)

        # NOTE: target is the POOLED feature vector, not raw x.
        target = self.pool_flatten(x)
        recon_loss = criterion(x_recon, target)

        if self.d_aux and aux is not None:
            aux_loss = criterion(aux_recon, aux)
        else:
            aux_loss = x.new_tensor(0.)

        constraint_loss = x.new_tensor(0.)
        if alpha > 0:
            B, D = hidden.shape
            M = hidden.t() @ hidden
            C = C_factor * torch.eye(D, device=hidden.device) - M
            constraint_loss = alpha * torch.norm(C) / (B * D)

        (recon_loss + beta * aux_loss + constraint_loss).backward()
        optimizer.step()
        return recon_loss.item(), aux_loss.item()


def load_ae_model(ae_model: nn.Module, full_checkpoint_path: str, device: torch.device) -> nn.Module:
    state_dict = torch.load(full_checkpoint_path, map_location=device or 'cpu')
    ae_model.load_state_dict(state_dict)
    ae_model.eval()
    return ae_model


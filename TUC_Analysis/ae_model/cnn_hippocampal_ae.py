import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional


class Conv_AE(nn.Module):
    """
    Adapt from Denis. Only with an additional 1x1 conv layer to reduce the input channels from 512 to 32, and a linear output layer without sigmoid activation to allow for unbounded reconstructions.
    (Feature map should be in [0 , 1] either way.)
    """
    def __init__(
        self, 
        n_hidden: int, 
        last_layer_activation: Optional[nn.Module] = None,
        d_aux: Optional[int] = None
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.d_aux = d_aux if d_aux is not None else 0
        self.last_layer_activation = last_layer_activation

        self._C, self._H, self._W = 8, 8, 11   # 8 * 8 * 11 = 704
        self._flat = self._C * self._H * self._W

        # Encoder
        self.enc_1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.enc_c1  = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)  # 32x16x21
        self.enc_c2  = nn.Conv2d(32,  8, kernel_size=3, stride=2, padding=1)  # 8x8x11
        self.enc_fc  = nn.Linear(self._flat + self.d_aux, n_hidden)

        # Decoder — feature path
        self.dec_fc  = nn.Linear(n_hidden, self._flat)
        self.dec_c1  = nn.ConvTranspose2d(8,  32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0))
        self.dec_c2  = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 0))
        self.dec_1x1 = nn.Conv2d(64, 512, kernel_size=1)

        # Decoder — aux head
        if self.d_aux:
            self.dec_aux = nn.Linear(n_hidden, self.d_aux)

    def encoder(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None):
        x = F.relu(self.enc_1x1(x))
        x = F.relu(self.enc_c1(x))
        x = F.relu(self.enc_c2(x))
        x = x.view(x.size(0), -1)              # B x 704
        if aux is not None:
            x = torch.cat([x, aux], dim=1)     # B x (704 + d_aux)
        return F.relu(self.enc_fc(x))

    def decoder(self, h: torch.Tensor):
        x = F.relu(self.dec_fc(h))
        x = x.view(x.size(0), self._C, self._H, self._W)
        x = F.relu(self.dec_c1(x))
        x = F.relu(self.dec_c2(x))
        x_recon   = self.dec_1x1(x)
        aux_recon = self.dec_aux(h) if self.d_aux > 0 else None
        if self.last_layer_activation is not None:
            x_recon = self.last_layer_activation(x_recon)
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
        beta: float = 1.0 
    ):    
        optimizer.zero_grad()
        x_recon, aux_recon, hidden = self.forward(x, aux)

        recon_loss = criterion(x_recon, x)
        
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



def load_ae_model(ae_model: Conv_AE, full_checkpoint_path: str, device: torch.device) -> Conv_AE:
    state_dict = torch.load(full_checkpoint_path, map_location=device or 'cpu')
    ae_model.load_state_dict(state_dict)
    ae_model.eval()
    return ae_model


import torch
import torch.nn as nn
import torch.nn.functional as F

# Auto‑encoder with balanced one‑hot latent code
class HippAE(nn.Module):
    '''
    One-hidden layer autoencoder with a softmax activation function in latent space that has a tuneable temperature (akin to controlling
    global inhibition / competition).
    '''
    def __init__(self, input_size, bottleneck_size, temperature_start):
        super().__init__()
        self.encoder = nn.Linear(input_size, bottleneck_size, bias=True)
        self.decoder = nn.Linear(bottleneck_size, input_size, bias=True)
        self.temperature = temperature_start

    def forward(self, x):
        logits = self.encoder(x)
        z = F.softmax(logits / self.temperature, dim=-1)
        x_rec = self.decoder(z)
        return x_rec, z


# Losses
def balance_loss(z):
    '''
    promotes uniformity of usage (each latent encodes approx 1/k samples), and embodies the idea of long-term homeostatic plasticity in Hipp.
    '''
    mean_code = z.mean(dim=0)
    uniform_target = torch.full((z.size(1),), 1.0 / z.size(1), device=z.device)
    return F.mse_loss(mean_code, uniform_target, reduction='sum')


def entropy_loss(z, eps=1e-10):
    '''
    promotes one-hot encodings, and embodies the idea of massive decorrelation and sparsity in Hipp.
    '''
    ent = -(z * (z + eps).log()).sum(dim=1).mean()
    return ent
    
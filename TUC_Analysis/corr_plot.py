import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Optional, Sequence
from utils import build_dataloader, StratifiedPoseSampler
from grid_cells.encoder import GridCellEncoder, GridModule, save_grid_encoder, load_grid_encoder
from main import load_feature_extractor, build_model_from_config
from main import load_run_config, build_model_from_config
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

class StratifiedPoseEmptier(StratifiedPoseSampler):
    """
    Bins the poses into the same bins as the Sampler but empties one bin after
    the other instead of round-robin.
    """
    def __init__(self, xy_bin_size: float, n_angle_bins: int = 8, seed: int = 0):
        super().__init__(xy_bin_size, n_angle_bins, seed)

    def rank(self, df: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(df))
        b = self.bins(df)[perm]
        return perm[np.lexsort((rng.random(len(df)), b))]

SUBMOD = 0.5
CHECKPOINT_PATH = f"./ae_model/feature_extractor_ae_checkpoint/grid_pool2x2_att_subset{SUBMOD}/"
DATA_CSV = "../Denis/HIP_AE_VISUAL/Datasets/Tmaze_2/data.csv"
XYBINS = 0.15

def get_encoder_activity(device, feature_extractor, grid_cell_encoder, load=False, batch_size=128):
    """ Encode the full dataset on the encoder with and without attention"""
    tf = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((247, 327)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    loader = build_dataloader(
        DATA_CSV,
        transform=tf,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        seed=None,
    )
    if load and os.path.exists(CHECKPOINT_PATH + "/latent_unatt.pt"):
        latent_space = torch.load(CHECKPOINT_PATH + "/latent_unatt.pt")
        latent_space_att = torch.load(CHECKPOINT_PATH + "/latent_att.pt")
        print("Loading encoded latent space")
        return latent_space, latent_space_att, loader

    with torch.no_grad():
        ae_model, _, _ = build_model_from_config(checkpoint_dir=CHECKPOINT_PATH, device=device, weights_name="best_model_aux_input.pt")
        ae_model.pool.A_raw = nn.Parameter(torch.full((1, 512, 1, 1), float("-inf")).to(device))
        ae_model_att, _, _ = build_model_from_config(checkpoint_dir=CHECKPOINT_PATH, device=device, weights_name="best_model_aux_input.pt")
        print("Encoding")
        latent_space = []
        latent_space_att = []
        for inputs, xy in tqdm(loader):
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            aux = grid_cell_encoder(xy).to(device)
            latent_space.append(ae_model.encoder(features, aux))
            latent_space_att.append(ae_model_att.encoder(features, aux))
        torch.save(latent_space, CHECKPOINT_PATH + "/latent_unatt.pt")
        torch.save(latent_space_att, CHECKPOINT_PATH + "/latent_att.pt")

    return latent_space, latent_space_att, loader


def run(train_decoder=False, batch_size=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """ Main Function. Encode, Train and Plot. """
    feature_extr = load_feature_extractor("./attention_model/SAM_weights/")
    feature_extr = feature_extr.to(device)

    encoder = load_grid_encoder(os.path.join(CHECKPOINT_PATH, "grid_encoder.pt"))
    ## Generate cell responses taken with and without attention
    # Load encoded features if possible
    lat, lat_att, dataloader = get_encoder_activity(device, feature_extr, encoder, True, batch_size=batch_size)

    sampler = StratifiedPoseEmptier(XYBINS, 1)
    subset_size = 1.
    fullidx = sampler.select(pd.read_csv(DATA_CSV), subset_size)

    n_samples_total = (len(lat) - 1) * len(lat[0]) + len(lat[-1])
    batch_size = 128

    def to_image_shape(tensor):
        return tensor.detach().cpu().numpy().transpose((0, 2, 3, 1))

    ## Reconstruct some samples
    with torch.no_grad():
        lat_att = torch.cat(lat_att, dim=0)
        A = lat_att[fullidx]
        A = A[A.sum(dim=1) != 0]
        ccmat = torch.corrcoef(A)
    plt.imshow(ccmat.cpu().numpy())
    plt.ylabel("Non-zero grid cell responses")
    plt.savefig(f'{CHECKPOINT_PATH}/cross_corr_matrix_{SUBMOD}_{int(XYBINS*100)}.png',dpi=300, bbox_inches='tight')

    print(f"Saved image at {CHECKPOINT_PATH}")

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--train-decoder", action="store_true", help="Train the decoder model for image reconstruction")
    args.add_argument("--batch-size", type=int, default=128, help="Batch size for encoding and training")
    args.add_argument("--device", type=str, default="cuda", help="Device to use for computation (cuda or cpu)")
    args = args.parse_args()
    train_decoder = args.train_decoder
    batch_size = args.batch_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run(train_decoder, batch_size=batch_size, device=device)

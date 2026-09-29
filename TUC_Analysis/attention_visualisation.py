import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Optional, Sequence
from utils import build_dataloader
from main import load_feature_extractor, build_model_from_config
from main import load_run_config, build_model_from_config
import matplotlib.pyplot as plt
import matplotlib
import os

CHECKPOINT_PATH = "./ae_model/feature_extractor_ae_checkpoint/features_only_pool2x2_att/"
class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: Sequence[int]):
        ######
        # Decoder model for input reconstruction
        # Each input to the decoder is an array/vector/list of neuronal responses
        # of the size latent_dim
        #####
        # output_dim is the target size of the network output
        # take a quarter of it to initialize the first Conv2DTranspose layer
        # and to reduce the number of needed parameters in the network
        # we increase the spatial dimension two times later to the original size

        super().__init__()
        c, w, h = output_dim # 2 spatial dimension (w x h) and the feature-map/channel dimension
        w = int(w/4)
        h = int(h/4)
        d = 25 # should be half of number feature maps, as features are encoded very sparsely

        self.decoder = nn.Sequential(
            # first dense layer after input
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, d*w*h),
            nn.ReLU(),
            nn.Unflatten(1, (d, w, h)),
            nn.Dropout(0.2),
            # increase the spatial size by two
            nn.ConvTranspose2d(d, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            # increase the spatial size by two again to get final resolution
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            # output layer maps to a 3-channel RGB image
            nn.ConvTranspose2d(128, 3, kernel_size=3),
            nn.Sigmoid(),
        )


    def forward(self, x):
        return self.decoder(x)

    def training_step(self, optimizer, criterion, x, yb):
        optimizer.zero_grad()
        pred = self.forward(x)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        return loss


def train(decoder, device, feature_extractor, epochs=50):
    """ Train the decoder with batches from the unattented encoder. """
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    tf = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((247, 327)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    loader = build_dataloader(
        "../Denis/HIP_AE_VISUAL/Datasets/Tmaze_2/data.csv",
        transform=tf,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        seed=None,
    )

    ae_model, _, _ = build_model_from_config(checkpoint_dir=CHECKPOINT_PATH, device=device, weights_name="best_model.pt")
    #train without attention
    ae_model.pool.A_raw = nn.Parameter(torch.full((1, 512, 1, 1), float("-inf")).to(device))
    decoder.train()
    decoder = decoder.to(device)
    epoch_loss = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        loss = 0
        for inp, target in tqdm(loader, leave=False):
            inp = inp.to(device, non_blocking=True)
            with torch.no_grad():
                features = feature_extractor(inp)
                x = ae_model.encoder(features)
            # x = torch.stack([latent[i//512][i%512] for i in idx]).to(device)
            loss += decoder.training_step(optimizer, criterion, x=x, yb=inp)
        epoch_loss.append(loss / len(loader))
        pbar.set_postfix(loss=f'{epoch_loss[-1]:.4f}')
        torch.save({
            "epoch": epoch,
            "loss": epoch_loss,
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, f'{CHECKPOINT_PATH}/decoder_ckpt.pt')
    return epoch_loss


def get_encoder_activity(device, feature_extractor, load=False):
    """ Encode the full dataset on the encoder with and without attention"""
    tf = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((247, 327)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    loader = build_dataloader(
        "../Denis/HIP_AE_VISUAL/Datasets/Tmaze_2/data.csv",
        transform=tf,
        batch_size=256,
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
        ae_model, _, _ = build_model_from_config(checkpoint_dir=CHECKPOINT_PATH, device=device, weights_name="best_model.pt")
        ae_model.pool.A_raw = nn.Parameter(torch.full((1, 512, 1, 1), float("-inf")).to(device))
        ae_model_att, _, _ = build_model_from_config(checkpoint_dir=CHECKPOINT_PATH, device=device, weights_name="best_model.pt")
        print("Encoding")
        latent_space = []
        latent_space_att = []
        for inputs, _ in tqdm(loader):
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            latent_space.append(ae_model.encoder(features))
            latent_space_att.append(ae_model_att.encoder(features))
        torch.save(latent_space, CHECKPOINT_PATH + "/latent_unatt.pt")
        torch.save(latent_space_att, CHECKPOINT_PATH + "/latent_att.pt")

    return latent_space, latent_space_att, loader


def run(train_decoder=False):
    """ Main Function. Encode, Train and Plot. """
    n_cells = 200
    output_dim = (128, 248, 328)
    # output_dim = (128, 31, 41)

    if not os.path.exists(f'{CHECKPOINT_PATH}/decoderimg/'):
        os.mkdir(f'{CHECKPOINT_PATH}/decoderimg/')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extr = load_feature_extractor("./attention_model/SAM_weights/")
    feature_extr = feature_extr.to(device)



    ## Create or Load a decoder model
    if train_decoder:
        # Train new decoder
        decoder_model = Decoder(n_cells, output_dim)
        decoder_model = decoder_model.to(device)
        print("Training decoder for image reconstruction")
        losstrace = train(decoder_model, device, feature_extr, epochs=200)

        # Save the weights
        torch.save(decoder_model, f'{CHECKPOINT_PATH}/decoder_model.pt')

        plt.figure()
        plt.semilogy(losstrace)
        plt.savefig(f'{CHECKPOINT_PATH}/losses.png')
    else:
        print("Loading pretrained decoder")
        decoder_model = torch.load(f'{CHECKPOINT_PATH}/decoder_model.pt', weights_only=False)

    ## Generate cell responses taken with and without attention
    # Load encoded features if possible
    lat, lat_att, dataloader = get_encoder_activity(device, feature_extr, True)

    n_samples_total = (len(lat) - 1) * len(lat[0]) + len(lat[-1])
    n_samples = 8

    print(f"Take {n_samples} Samples from {n_samples_total}")

    views, _ = next(iter(dataloader))
    idx = torch.randperm(len(views))[:n_samples]
    views = views[idx].to(device)

    def to_image_shape(tensor):
        return tensor.detach().cpu().numpy().transpose((0, 2, 3, 1))

    ## Reconstruct some samples
    ln = torch.stack((*lat[0][idx], *lat_att[0][idx]))
    with torch.no_grad():
        FE_out = to_image_shape(feature_extr(views))
        pred = decoder_model(ln)

    pred, pred_att = to_image_shape(pred[:n_samples]), to_image_shape(pred[n_samples:])
    views = (to_image_shape(views) * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    views = (views * 255).astype('uint8')

    fig, axs = plt.subplots(n_samples, 4,  figsize=(8,12))
    titles = ['Scene', 'No attention', 'Full attention' , 'Feat Extractor out']

    for i in range(n_samples):
        axs[i,0].imshow(views[i])
        axs[i,1].imshow(pred[i])
        axs[i,2].imshow(pred_att[i], interpolation='none')
        axs[i,3].imshow(np.sum(FE_out[i],axis=-1), interpolation='none', cmap='inferno')
        for n in range(4):
            if i==0:
                axs[i, n].set_title(titles[n])
            axs[i, n].set_axis_off()

    fig.savefig(f'{CHECKPOINT_PATH}/decoder_examples.png',dpi=300, bbox_inches='tight')
    print(f"Saved image at {CHECKPOINT_PATH}")

if __name__ == "__main__":
    run()

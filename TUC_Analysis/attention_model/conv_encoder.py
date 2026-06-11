import torch
import torch.nn as nn
import torch.nn.functional as F


###
# We define here only the encoder part
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder with strides to downsample the feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1
            ),  # Downsample by 2
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),  # Downsample by 2
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),  # Downsample by 2
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ReLU(),
        )

        # Concept projection
        self.concept_proj = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.Softmax(dim=1),
        )

    def forward(self, batch):
        # Encoder
        encoded_features = self.encoder(batch)
        # Concept projection
        concept_features = self.concept_proj(encoded_features)
        return concept_features


if __name__ == "__main__":
    import torch

    encoder = ConvEncoder()
    batch = torch.rand(1, 3, 120, 160)
    concept_features = encoder(batch)
    print(concept_features.shape)

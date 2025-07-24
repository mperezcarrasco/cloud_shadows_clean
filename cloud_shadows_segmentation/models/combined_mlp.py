import os
import torch
import torch.nn as nn
from typing import List
from models.unet import Unet
from models.scan import SpectralChannelAttentionNetwork


class CombinedModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        unet_model: nn.Module,
        san_model: nn.Module,
        mlp_dims: List[int] = [256, 128],
        dropout: float = 0.2,
    ):
        """
        Combines U-Net and SAN models using an MLP to merge their predictions.

        Args:
            in_dim: Number of input channels/dimensions
            num_classes: Number of output classes
            unet_model: Pretrained U-Net model
            san_model: Pretrained SAN model
            mlp_dims: List of hidden dimensions for the MLP merger
            dropout: Dropout rate for the MLP
        """
        super().__init__()

        # Load pretrained models
        self.unet = unet_model
        self.san = san_model

        # Freeze the pretrained models
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.san.parameters():
            param.requires_grad = False

        # Create MLP for combining predictions
        layers = []
        prev_dim = num_classes * 2  # Concatenated predictions from both models

        for dim in mlp_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the combined model.

        Args:
            x: Input tensor of shape (B, H, W, C)
        """
        # Preprocess data for each model
        x_unet, _ = self.unet.preprocess(x, None)
        x_san, _ = self.san.preprocess(x, None)

        # Get predictions from both models
        with torch.no_grad():
            unet_pred = self.unet(x_unet)  # Shape: (B, C, H, W)
            san_pred = self.san(x_san)  # Shape: (B, H, W, C)

        # Align dimensions
        unet_pred = unet_pred.permute(0, 2, 3, 1)  # Convert to (B, H, W, C)

        # Reshape predictions for MLP
        batch_size, height, width, num_classes = san_pred.shape
        unet_flat = unet_pred.reshape(-1, num_classes)
        san_flat = san_pred.reshape(-1, num_classes)

        # Combine predictions
        combined = torch.cat([unet_flat, san_flat], dim=1)

        # Pass through MLP
        merged = self.mlp(combined)

        # Reshape back to original dimensions
        output = merged.reshape(batch_size, height, width, num_classes)

        return output

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        """
        Calculate loss following the same pattern as individual models.
        """
        logits = self.forward(x)
        # Reshape logits and y for loss calculation
        B, H, W, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        y_flat = y.reshape(-1)

        loss = nn.functional.cross_entropy(
            logits_flat, y_flat, weight=class_weights, reduction=reduction
        )

        if return_logits:
            return loss, logits
        return loss

    def predict(self, x):
        """Get the mask prediction."""
        return torch.argmax(self(x), dim=-1)

    def preprocess(self, x, y):
        """Preprocess the data - returns data in (B, H, W, C) format."""
        return x, y


def create_combined_model(in_dim: int, num_classes: int, fold: int):
    """
    Creates and initializes the combined model with pretrained weights.
    """

    unet_path = f"../experiments/msat_cs_filtered_with_low_signal/unetv1_lr5e-3_none_wTrue_f{fold}/checkpoint_best.pth"
    san_path = f"../experiments/msat_cs_filtered_with_low_signal/improvedhlr_lr1e-3_none_wTrue_f{fold}/checkpoint_best.pth"
    # unet_path = f"../experiments/exp_full/unetv1_lr1e-3_std_full_wTrue_f{fold}/checkpoint_best.pth"
    # san_path = f"../experiments/exp_full/improvedhlr_lr1e-2_std_full_wTrue_f{fold}/checkpoint_best.pth"
    # Initialize individual models
    unet = Unet(in_dim=in_dim, num_classes=num_classes)
    san = SpectralChannelAttentionNetwork(in_dim=in_dim, num_classes=num_classes)

    # Load pretrained weights
    unet.load_state_dict(torch.load(unet_path)["model"])
    san.load_state_dict(torch.load(san_path)["model"])

    # Set models to evaluation mode
    unet.eval()
    san.eval()

    # Create combined model
    combined_model = CombinedModel(
        in_dim=in_dim, num_classes=num_classes, unet_model=unet, san_model=san
    )

    return combined_model

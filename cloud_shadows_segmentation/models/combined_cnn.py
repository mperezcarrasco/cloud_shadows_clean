import os
import torch
import torch.nn as nn
from typing import List, Tuple
from models.unet import Unet
from hyperspectral_artifact_removal.models.scan import SpectralChannelAttentionNetwork


class CombinedModelCNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        unet_model: nn.Module,
        san_model: nn.Module,
        cnn_channels: List[int] = [64, 32, 16],
        dropout: float = 0.2,
    ):
        """
        Combines U-Net and SAN models using a CNN to merge their predictions.

        Args:
            in_dim: Number of input channels/dimensions
            num_classes: Number of output classes
            unet_model: Pretrained U-Net model
            san_model: Pretrained SAN model
            cnn_channels: List of channel dimensions for the CNN merger
            dropout: Dropout rate for the CNN
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

        # Create CNN for combining predictions
        # Input will have 2*num_classes channels (concatenated predictions)
        cnn_layers = []
        in_channels = num_classes * 2  # Concatenated predictions from both models

        # Add convolutional layers
        for out_channels in cnn_channels:
            cnn_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        # Final layer to produce class predictions
        cnn_layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))

        self.cnn = nn.Sequential(*cnn_layers)
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass of the combined model with CNN merger.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            output: Output tensor of shape (B, H, W, num_classes)
        """
        # Preprocess data for each model
        x_unet, _ = self.unet.preprocess(x, None)
        x_san, _ = self.san.preprocess(x, None)

        # Get predictions from both models
        with torch.no_grad():
            unet_pred = self.unet(x_unet)  # Shape: (B, C, H, W)
            san_pred = self.san(x_san)  # Shape: (B, H, W, C)

        # Convert SAN predictions to channel-first format for CNN
        san_pred = san_pred.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)

        # Concatenate along the channel dimension
        combined = torch.cat([unet_pred, san_pred], dim=1)  # Shape: (B, 2*C, H, W)

        # Pass through CNN merger
        merged = self.cnn(combined)  # Shape: (B, C, H, W)

        # Convert back to the expected output format (B, H, W, C)
        output = merged.permute(0, 2, 3, 1)

        return output

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        """
        Calculate loss for the combined model.
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


class CombinedModelMultiScaleCNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        unet_model: nn.Module,
        san_model: nn.Module,
        base_channels: int = 64,
        dropout: float = 0.2,
    ):
        """
        Combines U-Net and SAN models using a multi-scale CNN with skip connections
        to merge their predictions.

        Args:
            in_dim: Number of input channels/dimensions
            num_classes: Number of output classes
            unet_model: Pretrained U-Net model
            san_model: Pretrained SAN model
            base_channels: Base number of channels for the CNN
            dropout: Dropout rate for the CNN
        """
        super().__init__()

        # Load pretrained models
        self.unet = unet_model
        self.san = san_model
        self.num_classes = num_classes

        # Freeze the pretrained models
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.san.parameters():
            param.requires_grad = False

        # Initial convolution to process concatenated features
        self.init_conv = nn.Sequential(
            nn.Conv2d(num_classes * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels),
        )

        # Downsampling path
        self.down1 = self._make_down_block(base_channels, base_channels * 2, dropout)
        self.down2 = self._make_down_block(base_channels * 2, base_channels * 4, dropout)

        # Upsampling path with skip connections
        self.up1 = self._make_up_block(base_channels * 4, base_channels * 2, dropout)
        self.up2 = self._make_up_block(base_channels * 2, base_channels, dropout)

        # Final convolution to produce class predictions
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _make_down_block(self, in_channels, out_channels, dropout):
        """Create a downsampling block for the multi-scale CNN."""
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def _make_up_block(self, in_channels, out_channels, dropout):
        """Create an upsampling block with skip connections for the multi-scale CNN."""
        return nn.ModuleDict(
            {
                "upsample": nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                "conv": nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels),
                ),
            }
        )

    def forward(self, x):
        """
        Forward pass of the combined model with multi-scale CNN merger.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            output: Output tensor of shape (B, H, W, num_classes)
        """
        # Preprocess data for each model
        x_unet, _ = self.unet.preprocess(x, None)
        x_san, _ = self.san.preprocess(x, None)

        # Get predictions from both models
        with torch.no_grad():
            unet_pred = self.unet(x_unet)  # Shape: (B, C, H, W)
            san_pred = self.san(x_san)  # Shape: (B, H, W, C)

        # Convert SAN predictions to channel-first format
        san_pred = san_pred.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)

        # Concatenate along the channel dimension
        combined = torch.cat([unet_pred, san_pred], dim=1)  # Shape: (B, 2*C, H, W)

        # Apply multi-scale CNN with skip connections
        x1 = self.init_conv(combined)

        # Downsampling path
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Upsampling path with skip connections
        x = self.up1["upsample"](x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1["conv"](x)

        x = self.up2["upsample"](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2["conv"](x)

        # Final convolution
        x = self.final_conv(x)

        # Convert back to the expected output format (B, H, W, C)
        output = x.permute(0, 2, 3, 1)

        return output

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        """
        Calculate loss for the combined model.
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


def create_combined_model_cnn(in_dim: int, num_classes: int, fold: int, model_type: str = "cnn"):
    """
    Creates and initializes the combined model with CNN merger and pretrained weights.

    Args:
        in_dim: Number of input channels/dimensions
        num_classes: Number of output classes
        fold: Cross-validation fold number
        model_type: Type of merger to use, either "cnn" for simple CNN or
                   "multiscale" for multi-scale CNN with skip connections
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

    # Create combined model with selected merger type
    if model_type == "cnn":
        combined_model = CombinedModelCNN(
            in_dim=in_dim, num_classes=num_classes, unet_model=unet, san_model=san
        )
    elif model_type == "multiscale":
        combined_model = CombinedModelMultiScaleCNN(
            in_dim=in_dim, num_classes=num_classes, unet_model=unet, san_model=san
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return combined_model

import torch
import torch.nn as nn
from typing import List
from models.mlp_utils import LinearBlock


class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Assuming x is of shape (B, H, W, C)
        y = torch.mean(x, dim=(1, 2))  # Average over spatial dimensions
        y = self.fc(y).unsqueeze(1).unsqueeze(2)
        return x * y


class SpectralChannelAttentionNetwork(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        mlp_dims: List[int] = [20, 20],
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp_dims = mlp_dims
        self.spectral_attention = SpectralAttention(in_dim)

        if mlp_dims is not None:
            layers = []
            for dim_in, dim_out in zip([in_dim] + mlp_dims[:-1], mlp_dims):
                layers += [
                    LinearBlock(
                        dim_in,
                        dim_out,
                        drop=dropout,
                        bias=bias,
                    )
                ]

            self.linear = nn.Sequential(*layers)
            self.linear.append(nn.Linear(mlp_dims[-1], num_classes, bias=bias))
        else:
            self.proj = nn.Linear(in_dim + 2, num_classes, bias=bias)

        self.logistic = nn.Linear(num_classes, num_classes, bias=bias)

    def forward(self, x):
        # x is of shape (B, H, W, C)
        x, (mean, std) = self.m_std_norm(x, dim=-1)
        x = self.spectral_attention(x)

        if self.mlp_dims is not None:
            # Reshape to (B*H*W, C) for linear layers
            B, H, W, C = x.shape
            x = x.reshape(-1, C)
            output = self.linear(x)
            output = output.reshape(B, H, W, -1)
        else:
            z = self.proj(torch.cat([x, mean.expand_as(x), std.expand_as(x)], dim=-1))
            output = self.logistic(z)

        return output

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        logits = self.forward(x)
        # Reshape logits and y for loss calculation
        B, H, W, C = logits.shape
        logits = logits.reshape(-1, C)
        y = y.reshape(-1)
        loss = nn.functional.cross_entropy(logits, y, weight=class_weights, reduction=reduction)

        if return_logits:
            return loss, logits.reshape(B, H, W, C)
        return loss

    def predict(self, x):
        return torch.argmax(self(x), dim=-1)

    def preprocess(self, x, y):
        # x is already in shape (B, H, W, C), no need to reshape
        return x, y

    def m_std_norm(self, x, dim=-1, eps=1e-5):
        mean_ = torch.mean(x, dim=dim, keepdim=True)
        std_ = torch.std(x, dim=dim, correction=0, keepdim=True)
        x = (x - mean_) / (std_ + eps)
        return x, (mean_, std_)

import torch
import torch.nn as nn

from typing import List

from models.mlp_utils import LinearBlock


class HyperspectralLogisticRegressionModel(nn.Module):
    """Hyperspectral Logistic Regression (HSR) Model.

    This model is based on the implementation of `Hyperspectral shadow removal
    with Iterative Logistic Regression and latent Parametric
    Linear Combination of Gaussians <https://arxiv.org/html/2312.15386v1>`_.


    Args:
        in_dim (int): Input dimension (i.e. spectral channels to be passed through the model).
        mlp_dims (list[int]): Hidden layer dimenisons of the MLP backbone. If none, MLP is not used. Default: None
        bias (bool): If use bias in the linear layers. Default: True
        dropout (float): Dropout rate if MLP is used. Default: 0.0
        lambda_l1 (float): l1 regularization hyperameter. Default: 0.0
        lambda_l2 (float): l2 regularization hyperameter. Default: 0.0
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        mlp_dims: List[int,] = [20, 20],
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp_dims = mlp_dims

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
            self.linear = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, x):
        """Forward function."""
        x, (mean, std) = self.m_std_norm(x, dim=-1)
        output = self.linear(x)
        return output

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        """Get the loss function for the HSR Model."""
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y, weight=class_weights, reduction=reduction)

        if return_logits:
            return loss, logits
        return loss

    def predict(self, x):
        """Get the mask prediction for the hyperspectral data."""
        return torch.argmax(self(x), dim=-1)

    def preprocess(self, x, y):
        """Preprocess the data from (B H W C) to (B*H*W C), where C is the spectral dimension. Default: 1024"""
        # resize
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        y = y.reshape(B, H * W)

        # flatten
        x = x.reshape(B * H * W, C)
        y = y.reshape(B * H * W)
        return x, y

    def m_std_norm(self, x, dim=-1, eps=1e-5):
        mean_ = torch.mean(x, dim=dim, keepdims=True)
        std_ = torch.std(x, dim=dim, correction=0, keepdims=True)
        x = (x - mean_) / (std_ + eps)
        return x, (mean_, std_)

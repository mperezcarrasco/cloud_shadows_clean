import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    """Pytorch implementation of a linear block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        act_fn: nn.Module = nn.LeakyReLU(0.1),
    ):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=bias),
            nn.BatchNorm1d(output_dim),
            act_fn,
            nn.Dropout(drop),
        )
        self.init_weights()

    def init_weights(self):
        w = (param.data for name, param in self.named_parameters() if "Linear" in name)
        for t in w:
            torch.nn.init.normal_(t, 0.0, 0.02)

    def forward(self, x):
        return self.block(x)

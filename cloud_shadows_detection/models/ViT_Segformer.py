import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Projector(nn.Module):
    """
    Linear Embedding module for SegFormer head
    """

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels) if norm_cfg is not None else None
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activate(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, feature_channels, num_classes, embedding_dim=256, dropout_ratio=0.1):
        super().__init__()
        self.in_channels = feature_channels

        # MLPs for different stages
        self.linear_c1 = Projector(input_dim=feature_channels[0], embed_dim=embedding_dim)
        self.linear_c2 = Projector(input_dim=feature_channels[1], embed_dim=embedding_dim)
        self.linear_c3 = Projector(input_dim=feature_channels[2], embed_dim=embedding_dim)
        self.linear_c4 = Projector(input_dim=feature_channels[3], embed_dim=embedding_dim)

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type="BN", requires_grad=True),
        )
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n = c1.shape[0]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP module with GELU activation."""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SegFormerViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,  # Single embedding dimension since we're not using stages
        depth: int = 12,  # Single depth value
        num_heads: int = 12,  # Single num_heads value
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # Calculate patch info
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.Hp = self.Wp = img_size // patch_size

        # Position embedding and class token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.Hp * self.Wp + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Single list of transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )

        # SegFormer head
        self.decode_head = SegFormerHead(
            feature_channels=[embed_dim] * 4,  # Same dimension for all features
            num_classes=num_classes,
            embedding_dim=256,
        )

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x):
        B = x.shape[0]

        # Initial patch embedding
        x = self.patch_embed(x)

        # Add position embedding and class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = x + self.pos_embed[:, 1:, :]
        x = torch.cat((cls_tokens, x), dim=1)

        # Store outputs from each block
        outputs = []

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
            outputs.append(x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, self.Hp, self.Wp))

        # Return specific intermediate features
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]

    def forward(self, x):
        # Get multi-scale features
        features = self.forward_features(x)

        # Apply SegFormer head
        x = self.decode_head(features)

        # Interpolate to original image size
        x = F.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        return x

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        logits = self.forward(x)
        # Reshape logits and y for loss calculation
        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
        y = y.reshape(-1)
        loss = nn.functional.cross_entropy(logits, y, weight=class_weights, reduction=reduction)
        if return_logits:
            return loss, logits.reshape(B, H, W, C)
        return loss

    def preprocess(self, x, y):
        """Preprocess the data from (B H W C) to (B C H W), where C is the spectral dimension.
        Default: 1024
        """
        # resize
        # x: (B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        return x, y

    def predict(self, x):
        """Get the mask prediction for the hyperspectral data."""
        return torch.argmax(self(x), dim=1)

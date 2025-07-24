from models.hyperspectral_logreg import HyperspectralLogisticRegressionModel
from models.scan import SpectralChannelAttentionNetwork
from models.combined_mlp import create_combined_model
from models.combined_cnn import create_combined_model_cnn
from models.ViT_Segformer import SegFormerViT
from models.unet import Unet


def build_network(
    model_name: str,
    in_dim: int,
    num_classes: int,
    fold=0,
    mlp_dims=None,
):
    """Builds the feature extractor and the projection head."""
    implemented_networks = (
        "ilr",
        "mlp",
        "unet",
        "scan",
        "combined_mlp",
        "combined_cnn",
        "segformervit",
    )
    assert model_name in implemented_networks

    if model_name == "ilr" or model_name == "mlp":
        model = HyperspectralLogisticRegressionModel(
            in_dim,
            num_classes,
            (None if mlp_dims == "none" else [int(dim) for dim in mlp_dims.split(",")]),
        )
    elif model_name == "scan":
        model = SpectralChannelAttentionNetwork(
            in_dim,
            num_classes,
            (None if mlp_dims == "none" else [int(dim) for dim in mlp_dims.split(",")]),
        )
    elif model_name == "unet":
        model = Unet(in_dim, num_classes)
    elif model_name == "segformervit":
        model = SegFormerViT(in_dim, num_classes)
    elif model_name == "combined_mlp":
        model = create_combined_model(in_dim, num_classes, fold)
    elif model_name == "combined_cnn":
        model = create_combined_model_cnn(in_dim, num_classes, fold)
    return model

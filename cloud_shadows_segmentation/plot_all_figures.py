import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import glob
import os

from models.build_model import build_network
from datasets.dataset import HyperspectralDataset
from torch.cuda.amp import autocast
import joblib

from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def format_lr(lr):
    return f"{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e')

model_name = 'ilr'
base_pth = '../data/mair_cs'
exp_pth = '../checkpoints/mair_cs'
norm_type = 'std_full'
hidden_dims = "none"
batch_size = 1
weighted = True
fold = 0
in_dim = 1024
lr = format_lr(1e-3)
mask_types = ["dark_surface_mask", "cloud_shadow_mask", "cloud_mask"]

def get_dataloader(
    base_path: str,
    norm_type: str,
    mask_types: List[str] = ["dark_surface_mask", "cloud_shadow_mask", "cloud_mask"],
    batch_size: int = 32,
    num_workers: int = 0,
    fold: int = 0,
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Build and return dataloaders for training and validation.
    
    Args:
        base_path (str): Base path for data files.
        norm_type (str): Type of normalization to apply.
        mask_types (List[str]): Types of masks to use. Default is ["dark_surface_mask", "cloud_shadow_mask", "cloud_mask"].
        model_name (str): Name of the model. Default is "logreg".
        batch_size (int): Batch size for dataloaders. Default is 32.
        num_workers (int): Number of worker processes for data loading. Default is 0.
        fold (int): Fold number for cross-validation. Default is 0.
    
    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, int]]: Train dataloader, validation dataloader, and full class map.
    """
    assert any(x in ["plumes", "cloud_shadow_mask", "cloud_mask", "dark_surface_mask"] for x in mask_types)
    
    obj_list = np.load(base_path + "/mask_list.npy").tolist()
    
    obj_list = obj_list
    scaler = None if norm_type in ["none", "l_norm_clip"] else joblib.load(f"{base_path}/scaler_{norm_type}.pkl")
    class_map = {type_: i + 1 for i, type_ in enumerate(mask_types)}  # label 0 reserved for normal objects

    dataset_type = "plumes" if "plumes" in mask_types else base_path.split("/")[-1]

    inference_dataset = HyperspectralDataset(
        base_path, obj_list, norm_type, class_map, scaler, dataset_type, apply_transforms=False, test=True,
    )
    inference_loader = DataLoader(
        inference_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )
    full_class_map = {**class_map, "background": 0}
    return inference_loader, full_class_map


def get_flat_preds(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    use_amp: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Obtaining flat predictions and labels from the model with optional mixed precision."""
    preds, labels = [], []

    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data = data.float().to(device)
            label = label.long().to(device)

            data, label = model.preprocess(data, label)
            
            with autocast(enabled=use_amp):
                pred = model.predict(data)
            
            pred = pred.detach().cpu().flatten()
            label = label.cpu().flatten()

            preds.append(pred)
            labels.append(label)

    return torch.cat(preds).numpy().astype(int), torch.cat(labels).numpy().astype(int)

def save_individual_image_plot(data, mask, pred, color_maps_rgb, inv_map, sample_idx, 
                              output_folder="predictions/", figsize=(10, 6)):
    """
    Creates and saves individual visualization plot for one image (data, mask, prediction).
    
    Args:
        data (torch.Tensor): Input data tensor
        mask (torch.Tensor): Ground truth mask tensor
        pred (torch.Tensor): Model prediction tensor
        color_maps_rgb (dict): Dictionary mapping class names to RGB color values (0-255)
        inv_map (dict): Dictionary mapping class indices to class names
        sample_idx (int): Index of the sample for filename
        output_folder (str): Path to folder where plot will be saved
        figsize (tuple): Figure size (width, height)
    """
    # Convert tensors to numpy
    data_np = data.cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    # Transpose for better visualization (rotate 90 degrees)
    #data_np = np.transpose(data_np)
    #mask_np = np.transpose(mask_np)
    #pred_np = np.transpose(pred_np)
    
    H, W = mask_np.shape
    
    # Create coordinate meshes for proper alignment
    Y, X = np.mgrid[0:H, 0:W]
    
    # Create custom colormap for categorical data
    colors_list = np.zeros((len(inv_map), 3))
    for idx, class_name in inv_map.items():
        colors_list[idx] = color_maps_rgb[class_name] / 255.0
    custom_cmap = ListedColormap(colors_list)
    
    # Create figure for this individual image
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Sample {sample_idx+1:03d}: Data, Mask, and Prediction', fontsize=16, y=1)
    
    # DATA PLOT
    axes[0].set_title('Data', fontsize=14)
    axes[0].pcolormesh(X, Y, data_np,
                      cmap="Greys_r",
                      shading='nearest',
                      rasterized=True)
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    
    # MASK PLOT
    axes[1].set_title('Mask', fontsize=14)
    axes[1].pcolormesh(X, Y, mask_np,
                      cmap=custom_cmap,
                      shading='nearest',
                      rasterized=True,
                      vmin=-0.5,
                      vmax=len(inv_map)-0.5)
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    
    # PREDICTION PLOT
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].pcolormesh(X, Y, pred_np,
                      cmap=custom_cmap,
                      shading='nearest',
                      rasterized=True,
                      vmin=-0.5,
                      vmax=len(inv_map)-0.5)
    axes[2].set_aspect('equal')
    axes[2].axis('off')
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Add legend at the bottom
    patches = []
    for label, color in color_maps_rgb.items():
        color_norm = np.array(color) / 255.0
        clean_label = label.split("_mask")[0].replace('_', ' ').title()
        patches.append(mpatches.Patch(color=color_norm, label=clean_label))
    
    fig.legend(handles=patches, 
              loc='lower center', 
              ncol=len(color_maps_rgb),
              fontsize=12,
              frameon=True,
              framealpha=0.8,
              bbox_to_anchor=(0.5, 0.02))
    
    # Save the figure
    filename = f"image_sample_{sample_idx+1:03d}.png"
    filepath = os.path.join(output_folder, filename)
    #plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Saved: {filepath}")

def create_individual_plots_efficiently(val_loader, model, device, model_name, color_maps_rgb, inv_map,
                                       output_folder="predictions/", max_samples=None):
    """
    Efficiently creates and saves individual plots for each image in the dataloader.
    
    Args:
        val_loader: PyTorch dataloader
        model: Trained model
        device: Device to run inference on
        model_name: Name of the model for handling different architectures
        color_maps_rgb: Color mapping for visualization
        inv_map: Inverse class mapping
        output_folder: Folder to save plots
        max_samples: Maximum number of samples to process (None for all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    model.eval()
    sample_idx = 0
    
    # Create custom colormap for categorical data
    colors_list = np.zeros((len(inv_map), 3))
    for idx, class_name in inv_map.items():
        colors_list[idx] = color_maps_rgb[class_name] / 255.0
    custom_cmap = ListedColormap(colors_list)
    
    with torch.no_grad():
        for data, mask in val_loader:
            # Stop if we've reached the maximum number of samples
            if max_samples is not None and sample_idx >= max_samples:
                break
                
            # Get batch dimensions
            B, H, W, C = data.shape
            
            # Process each image in the batch (though typically batch_size=1 for visualization)
            for b in range(B):
                if max_samples is not None and sample_idx >= max_samples:
                    break
                    
                # Extract single image and mask
                single_data = data[b:b+1]  # Keep batch dimension
                single_mask = mask[b:b+1]
                
                # Move to device and preprocess
                single_data = single_data.float().to(device)
                single_data, single_mask = model.preprocess(single_data, single_mask)
                
                # Get prediction
                pred = model.predict(single_data.float()).cpu()
                
                # Handle different model architectures
                if model_name in ['unetv1', 'unetfull', 'segformervit']:
                    # For models that expect channel-first input but return different formats
                    processed_data = single_data.permute(0, 2, 3, 1)[0].cpu()
                else:
                    # For models that work with flattened or reshaped data
                    processed_data = single_data.reshape(1, H, W, C)[0].cpu()
                
                # Reshape outputs
                processed_mask = single_mask.reshape(1, H, W)[0]
                processed_pred = pred.reshape(1, H, W)[0]
                
                # Extract the first channel for data visualization
                data_for_plot = processed_data[:, :, 0]
                
                # Save individual plot
                save_individual_image_plot(
                    data_for_plot, processed_mask, processed_pred, 
                    color_maps_rgb, inv_map, sample_idx, output_folder
                )
                
                sample_idx += 1
                
                if sample_idx % 10 == 0:
                    print(f"Processed {sample_idx} images...")
    
    print(f"Finished processing {sample_idx} images. All plots saved to {output_folder}")


if torch.cuda.is_available():
    device = "cuda:1"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_name_plot = {'ilr': "ILR", 
                   'logreg': "MLP", 
                   'unetv1': "UNet", 
                   "improvedhlr": "SAN", 
                   "combined": "CombinedMLP",
                   "combined_cnn": "CombinedCNN",
                  }


job_name = f"{model_name}_lr{lr}_{norm_type}_w{weighted}_f{fold}"
directory = os.path.join(exp_pth, job_name)

loader, class_map = get_dataloader(
    base_pth,
    norm_type,
    mask_types,
    batch_size=batch_size,
    num_workers=0,
    fold=fold,
)
num_classes = len(class_map)

color_maps_rgb = {"background": np.array([147,112,219]), #purple
                  "dark_surface_mask": np.array([0, 0, 255]), #blue
             "cloud_shadow_mask": np.array([0, 128, 0]), #green
             "cloud_mask": np.array([255,255,0])}  #yellow

inv_map = {value:key for key,value in class_map.items()}

model = build_network(
    model_name,
    in_dim,
    num_classes,
    fold,
    hidden_dims,
)
print(model)
cpkt = torch.load("{}/checkpoint_best.pth".format(directory))
model.load_state_dict(cpkt["model"])
model.to(device)

# Use this:
create_individual_plots_efficiently(
    loader, model, device, model_name, color_maps_rgb, inv_map,
    output_folder="../predictions/mair_cs/",
)
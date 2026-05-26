import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for safe execution
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv8 model feature maps")
    parser.add_argument("--model", default="models/best.pt", help="Path to model weights (default: models/best.pt)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--layer", type=int, default=2, help="Layer index to visualize (0 to 21, default: 2)")
    parser.add_argument("--max-maps", type=int, default=16, help="Maximum number of feature maps to display (default: 16)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[Error] Image not found at {args.image}")
        return

    # Load model
    print(f"[Info] Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"[Error] Failed to read image from {args.image}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to match YOLO standard input size
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(next(model.model.parameters()).device)

    # Get modules of the backbone
    modules = list(model.model.model.children())
    print(f"[Info] Total layers in model graph: {len(modules)}")
    
    if args.layer >= len(modules) or args.layer < 0:
        print(f"[Error] Layer index must be between 0 and {len(modules)-1}")
        return
        
    print(f"[Info] Extracting feature maps for layer {args.layer}: {modules[args.layer].__class__.__name__}")
    
    # Forward pass up to the selected layer
    x = img_tensor
    outputs = []
    
    # Custom partial forward pass
    with torch.no_grad():
        for i, module in enumerate(modules):
            # Standard YOLO layers forward logic
            # Some layers take multiple inputs or skip connections, but for early backbone layers (0-9) they are sequential
            try:
                x = module(x)
            except Exception:
                # Fallback for complex head modules
                print(f"[Warning] Sequential forward interrupted at layer {i} ({module.__class__.__name__}) due to head branching.")
                break
            if i == args.layer:
                outputs.append(x)
                break

    if not outputs:
        print("[Error] Failed to extract feature maps.")
        return

    features = outputs[0].squeeze(0).cpu().numpy()
    num_channels = features.shape[0]
    print(f"[Info] Feature maps shape: {features.shape} ({num_channels} channels)")
    
    # Grid dimensions for plotting
    n_display = min(args.max_maps, num_channels)
    cols = int(np.ceil(np.sqrt(n_display)))
    rows = int(np.ceil(n_display / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle(f"Layer {args.layer} Features ({modules[args.layer].__class__.__name__})", fontsize=16)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        ax.axis("off")
        if idx < n_display:
            f_map = features[idx]
            # Normalize to 0-1 for plotting
            f_map_norm = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-8)
            ax.imshow(f_map_norm, cmap="viridis")
            ax.set_title(f"Ch {idx}", fontsize=10)
            
    plt.tight_layout()
    output_path = f"runs/features_layer_{args.layer}.png"
    plt.savefig(output_path, dpi=150)
    print(f"[Success] Feature maps saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()

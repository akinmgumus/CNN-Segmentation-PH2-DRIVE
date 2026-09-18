# plot.py

import os
import torch
import matplotlib
matplotlib.use("Agg")  # required in headless environments such as HPC
import matplotlib.pyplot as plt

def visualize_prediction(model, dataset, device, idx, save_dir, prefix="sample"):
    """
    Save a 3-panel figure:
      - Original image
      - Ground truth mask
      - Predicted mask
    to: save_dir / f"{prefix}_{idx:03d}.png"
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 1) Take a sample from the dataset
    img, gt_mask = dataset[idx]
    img = img.to(device).unsqueeze(0)     # [1, C, H, W]
    gt_mask = gt_mask.squeeze().cpu().numpy()  # [H, W]

    # 2) Model prediction
    with torch.no_grad():
        logits = model(img)              # [1, 1, H, W]
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float().squeeze().cpu().numpy()  # [H, W]

    # 3) Move the image to CPU and [H,W,C] layout for display
    img_disp = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,C]

    # 4) Build the figure
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(img_disp)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(gt_mask, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Prediction")
    ax[2].axis("off")

    # 5) Save
    filename = os.path.join(save_dir, f"{prefix}_{idx:03d}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)  # free the memory

    # optional log line
    print(f"Saved: {filename}")

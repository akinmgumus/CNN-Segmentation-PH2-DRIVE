import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Datasets and modules
import PH2_Dataset as ph2
import DRIVE_Dataset as drive
from SimpleEncoderDecoder import SimpleEncoderDecoder
import metrics as M
import plot as p

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

learning_rate = 1e-3
batch_size = 32
num_epochs = 50

num_workers = 0
SEED = 42
# Set random seed for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss

# Transforms for images and masks
img_transform = transforms.Compose([
    transforms.Resize((128,128), interpolation=F.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((128,128), interpolation=F.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# Training function
def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        # Use BCEWithLogitsLoss for single-channel logits; masks should be (1, H, W)
        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"  Training Loss: {avg_loss:.6f}")

def check_metrics(loader, model, device=device):
    model.eval()

    all_metrics = {"dice": 0.0, "iou": 0.0, "accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            all_metrics["dice"]        += M.dice_coefficient(logits, masks).item()
            all_metrics["iou"]         += M.iou_score(logits, masks).item()
            all_metrics["accuracy"]    += M.accuracy_score(logits, masks).item()
            all_metrics["sensitivity"] += M.sensitivity_score(logits, masks).item()
            all_metrics["specificity"] += M.specificity_score(logits, masks).item()

    num_batches = len(loader)
    for key in all_metrics:
        all_metrics[key] /= num_batches

    print(f"  METRICS -> Dice: {all_metrics['dice']:.4f} | IoU: {all_metrics['iou']:.4f} | Acc: {all_metrics['accuracy']:.4f}")
    print(f"  Sens: {all_metrics['sensitivity']:.4f} | Spec: {all_metrics['specificity']:.4f}")

    return all_metrics


def main():
    """Main function. Runs training and evaluation for PH2 and DRIVE datasets."""
    # ======================================================================
    # SECTION 1: PH2
    # ======================================================================
    print("\n" + "="*30)
    print("  PH2 DATASET STARTING")
    print("="*30)

    # 1.1. Load PH2 dataset and split
    try:
        ph2_dataset = ph2.PH2(img_transform=img_transform, mask_transform=mask_transform)
        total_size = len(ph2_dataset)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size   

        train_set, val_set, test_set = torch.utils.data.random_split(ph2_dataset, [train_size, val_size, test_size])

        ph2_train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
        ph2_test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0)
        ph2_val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)

        
    except Exception as e:
        print(f"ERROR: Could not load PH2 data loaders. Check 'PH2' function.")
        print(e)
        return

    # 1.2. Model and optimizer
    model_ph2 = SimpleEncoderDecoder(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model_ph2.parameters(), lr=learning_rate)

    # 1.3. Training loop
    best_val_dice = -1.0
    best_ph2_state = None

    print("\nTraining PH2 model...")
    for epoch in range(num_epochs):
        print(f"\n--- PH2 Epoch {epoch+1}/{num_epochs} ---")
        train_one_epoch(ph2_train_loader, model_ph2, optimizer, loss_fn)

        print("PH2 Validation Metrics:")
        val_metrics = check_metrics(ph2_val_loader, model_ph2, device=device)
        val_dice = val_metrics["dice"]
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_ph2_state = model_ph2.state_dict()
            torch.save(best_ph2_state, 'cnn_encoder_decoder_ph2.pth')
    
    # 1.4. Final test
    print("\nPH2 training completed. Final test metrics:")
    check_metrics(ph2_test_loader, model_ph2, device=device)

    # Plot some predictions and save them into 'ph2' folder
    ph2_test_dataset = ph2_test_loader.dataset

    # 10 examples of predictions
    num_examples = min(10, len(ph2_test_dataset))

    for j in range(num_examples):
        p.visualize_prediction(
            model_ph2,
            ph2_test_dataset,
            device=device,
            idx=j,
            save_dir="ph2",
            prefix="ph2"
        )

    # ======================================================================
    # SECTION 2: DRIVE
    # ======================================================================
    print("\n" + "="*30)
    print("  STARTING DRIVE DATASET")
    print("="*30)

    # 2.1. Prepare DRIVE datasets and loaders
    try:
        drive_train_val_test = drive.DRIVE(img_transform=img_transform, mask_transform=mask_transform)
        
        total_size = len(drive_train_val_test)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size
    
        train_set, val_set, test_set = torch.utils.data.random_split(drive_train_val_test, [train_size, val_size, test_size])

        drive_train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
        drive_test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0)
        drive_val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)

    except Exception as e:
        print(f"ERROR: Could not load DRIVE data loaders. Check 'DRIVE' function.")
        print(e)
        return

    # 2.2. Model and optimizer
    model_drive = SimpleEncoderDecoder(in_channels=3, out_channels=1).to(device)
    optimizer_drive = optim.Adam(model_drive.parameters(), lr=learning_rate)

    # 2.3. Training loop
    print("\nTraining DRIVE model...")

    best_val_dice = -1.0
    best_drive_state = None

    for epoch in range(num_epochs):
        print(f"\n--- DRIVE Epoch {epoch+1}/{num_epochs} ---")
        train_one_epoch(drive_train_loader, model_drive, optimizer_drive, loss_fn)

        print("DRIVE Validation Metrics:")
        val_metrics = check_metrics(drive_val_loader, model_drive, device=device)
        val_dice = val_metrics["dice"]
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_drive_state = model_drive.state_dict()
            torch.save(best_drive_state, 'cnn_encoder_decoder_drive.pth')

    # 2.4. Final test
    
    print("\nDRIVE training completed. Final test metrics:")
    check_metrics(drive_test_loader, model_drive, device=device)

    # Plot some predictions and save them into 'drive' folder
    drive_test_dataset = drive_test_loader.dataset
    num_examples = min(10, len(drive_test_dataset))

    for i in range(num_examples):
        p.visualize_prediction(
            model_drive,
            drive_test_dataset,
            device=device,
            idx=i,
            save_dir="drive",
            prefix="drive"
        )


if __name__ == "__main__":
    main()
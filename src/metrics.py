import torch

def _ensure_channel_dim(x):
    if x.dim() == 3:   # [B, H, W]
        return x.unsqueeze(1)
    return x           # already [B, 1, H, W]

def binarize(preds, threshold=0.5):
    probs = torch.sigmoid(preds)
    return (probs > threshold).float()

def dice_coefficient(preds, targets, smooth=1e-6):
    preds = binarize(preds)
    preds   = _ensure_channel_dim(preds)
    targets = _ensure_channel_dim(targets)

    intersection = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(preds, targets, smooth=1e-6):
    preds = binarize(preds)
    preds   = _ensure_channel_dim(preds)
    targets = _ensure_channel_dim(targets)

    intersection = (preds * targets).sum(dim=(2,3))
    total = (preds + targets).sum(dim=(2,3))
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def accuracy_score(preds, targets):
    preds = binarize(preds)
    preds   = _ensure_channel_dim(preds)
    targets = _ensure_channel_dim(targets)

    correct = (preds == targets).float()
    return correct.sum() / correct.numel()

def sensitivity_score(preds, targets, smooth=1e-6):
    preds = binarize(preds)
    preds   = _ensure_channel_dim(preds)
    targets = _ensure_channel_dim(targets)

    TP = (preds * targets).sum(dim=(2,3))
    P = targets.sum(dim=(2,3))
    sens = (TP + smooth) / (P + smooth)
    return sens.mean()

def specificity_score(preds, targets, smooth=1e-6):
    preds = binarize(preds)
    preds   = _ensure_channel_dim(preds)
    targets = _ensure_channel_dim(targets)

    TN = ((1 - preds) * (1 - targets)).sum(dim=(2,3))
    N = (1 - targets).sum(dim=(2,3))
    spec = (TN + smooth) / (N + smooth)
    return spec.mean()

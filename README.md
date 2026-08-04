# CNN Encoder–Decoder Segmentation on PH2 and DRIVE

A baseline encoder–decoder for binary medical image segmentation, and what happens when you run the same architecture on two datasets that look superficially similar and are not: skin lesions in PH2, retinal blood vessels in DRIVE.

**Scope.** Four-person group project for **02516 Introduction to Deep Learning in Computer Vision** at DTU, autumn 2025. The full project also covered a U-Net, a loss-function ablation and weakly supervised segmentation from point clicks. **What is here is my part: the baseline encoder–decoder, the metrics, the plotting and the training loop.** The U-Net was Elif Pulukcu's and the weak-supervision work was Yusuf Eren Kilic's; neither is in this repository. Group: Akin Mert Gümüs, Burak Yenidzhe, Elif Pulukcu, Yusuf Eren Kilic. Original group repository: [ph2-drive-image-segmentation](https://github.com/yerenkl/ph2-drive-image-segmentation).

## The model

`SimpleEncoderDecoder` is deliberately plain — the point of a baseline is to be the thing a U-Net has to beat.

```
encoder      3×3 conv → ReLU → 2×2 maxpool     3 → 64      H×W    → H/2×W/2
             3×3 conv → ReLU → 2×2 maxpool    64 → 128    H/2×W/2 → H/4×W/4
bottleneck   3×3 conv → ReLU                 128 → 256
decoder      2×2 transposed conv, stride 2   256 → 128    H/4×W/4 → H/2×W/2
             3×3 conv → ReLU
             2×2 transposed conv, stride 2   128 → 64     H/2×W/2 → H×W
             3×3 conv → ReLU
             1×1 conv                         64 → 1      single-channel logits
```

The decoder mirrors the encoder, so spatial resolution comes back the way it went down. What it does **not** have is skip connections: nothing carries early, high-resolution features across to the decoder. Everything the output knows about fine detail has to survive the trip down to H/4×W/4 and back. That single omission is what the results are about.

Metrics are computed in `metrics.py` from a thresholded sigmoid at 0.5 — Dice, IoU, accuracy, sensitivity and specificity, all as tensor operations over the `[B, 1, H, W]` mask.

## What came out of it

On **PH2** — 200 dermoscopy images, one large well-defined lesion per image — the baseline works: **0.8864 Dice**, essentially matching the group's U-Net at 0.8935. When the object is big and roughly convex, skip connections buy you almost nothing, because there is no fine structure to preserve in the first place.

On **DRIVE** — 20 retinal images, vessels a few pixels wide, under 8 % of pixels in the foreground — it collapses: **0.1472 Dice**, against the U-Net's 0.7532.

The interesting part is *how* it collapses. The test metrics come out at sensitivity 1.0000 and specificity 0.0000, which is not a model that failed to find the vessels — it is a model that labelled **every pixel** a vessel. Faced with 20 training images and a foreground it could not resolve at H/4 resolution, the network found the degenerate solution that catches all the positives by giving up on the negatives entirely.

Put next to the U-Net's 0.7532 on the same data, that gives a fairly clean measurement of what skip connections are actually for: not general accuracy, but keeping thin structures alive through the bottleneck.

## Running it

```bash
pip install -r requirements.txt
python src/CNN_segment_main.py
```

Neither dataset is redistributed here. `CNN_segment_main.py` imports `PH2_Dataset` and `DRIVE_Dataset`, which are the group's shared loaders and live in the original repository — so this will not run end-to-end from this repository alone. It is published to show the model, the metric implementations and the training loop, not as a standalone package.

Splits are 70 % train / 15 % validation / 15 % test on both datasets.

## License

MIT — see [LICENSE](LICENSE).

"""
train_cnn_dose_features.py

Trains a small CNN on the whole-abdomen slices from extract_cnn_slices.py,
using your REAL ground-truth DLP (not a pseudo-label) as the training
target -- this is a genuine supervised task, at whole-scan granularity.

With only ~144 patients, training a CNN from scratch would badly overfit,
so this uses TRANSFER LEARNING: a ResNet18 pretrained on ImageNet, with
its early layers frozen and only the last block + a small regression head
fine-tuned. That's the standard, defensible approach for a dataset this
size -- say so explicitly in your report.

After training, it runs every patient/phase (train AND test -- we need
features for everyone, not just the training split) through the network
and saves the second-to-last layer's output as a feature vector per
patient/phase. Those features get merged into train_organ_dose_model.py
as EXTRA columns alongside organ_encoded/volume_cm3/mean_hu/mean_ctdivol_mGy
-- the CNN acts purely as a feature extractor for the RandomForest, per
your setup; it does not predict organ dose directly.

Usage:
    python train_cnn_dose_features.py --manifest cnn_slice_manifest.csv --slice-dir cnn_slices --dlp-labels "1 - 144  excel (2).xlsx" --out cnn_embeddings.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

EMBEDDING_DIM = 64

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_dlp_labels(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df["UID"] = df["UID"].astype(int).astype(str)

    plain = df[["UID", "DLP -PLAIN"]].rename(columns={"DLP -PLAIN": "real_dlp_mgycm"})
    plain["PHASE"] = "plain"
    venous = df[["UID", "DLP-VENOUS"]].rename(columns={"DLP-VENOUS": "real_dlp_mgycm"})
    venous["PHASE"] = "venous"
    return pd.concat([plain, venous], ignore_index=True)


class SliceDataset(Dataset):
    def __init__(self, df, slice_dir, train: bool):
        self.df = df.reset_index(drop=True)
        self.slice_dir = slice_dir
        # Light augmentation only for training -- this is a tiny dataset,
        # so a bit of augmentation helps, but keep it mild (real anatomy,
        # not a natural image -- no color jitter, no heavy crops).
        aug = [transforms.RandomHorizontalFlip(p=0.5)] if train else []
        self.transform = transforms.Compose(aug + [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1-channel CT -> 3-channel for ImageNet backbone
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.slice_dir}/{row['slice_path']}").convert("L")
        x = self.transform(img)
        y = torch.tensor(np.log1p(row["real_dlp_mgycm"]), dtype=torch.float32)
        return x, y


class DoseCNN(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze everything except the last residual block -- with ~140
        # training images, fine-tuning the whole network would overfit fast.
        for name, param in backbone.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop original fc
        self.embed = nn.Sequential(nn.Linear(512, embedding_dim), nn.ReLU())
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x, return_embedding=False):
        f = self.features(x).flatten(1)
        emb = self.embed(f)
        out = self.head(emb).squeeze(-1)
        if return_embedding:
            return out, emb
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--slice-dir", required=True)
    parser.add_argument("--dlp-labels", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="cnn_embeddings.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"UID": str})
    dlp = load_dlp_labels(args.dlp_labels)
    df = manifest.merge(dlp, on=["UID", "PHASE"], how="inner").dropna(subset=["real_dlp_mgycm"])
    print(f"{len(df)} patient/phase rows have both a slice and a real DLP label.")

    # Split by UID so the same patient never appears in both train and test
    # (their plain/venous rows would otherwise leak across the split).
    uids = list(df["UID"].unique())
    train_uids, test_uids = train_test_split(uids, test_size=0.2, random_state=42)
    train_df = df[df["UID"].isin(train_uids)]
    test_df = df[df["UID"].isin(test_uids)]
    print(f"Train: {len(train_df)} rows ({len(train_uids)} patients)   "
          f"Test: {len(test_df)} rows ({len(test_uids)} patients)")

    train_loader = DataLoader(SliceDataset(train_df, args.slice_dir, train=True),
                               batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(SliceDataset(test_df, args.slice_dir, train=False),
                              batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DoseCNN().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_df)

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    pred = model(x).cpu().numpy()
                    preds.extend(np.expm1(pred))
                    actuals.extend(np.expm1(y.numpy()))
            mae = mean_absolute_error(actuals, preds)
            r2 = r2_score(actuals, preds) if len(set(actuals)) > 1 else float("nan")
            print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
                  f"test_MAE={mae:.1f} mGy*cm  test_R2={r2:.3f}")

    # Extract embeddings for EVERY row (train + test), for downstream use
    # as RandomForest features.
    model.eval()
    full_loader = DataLoader(SliceDataset(df, args.slice_dir, train=False),
                              batch_size=args.batch_size, shuffle=False)
    all_embeddings = []
    with torch.no_grad():
        for x, _ in full_loader:
            x = x.to(device)
            _, emb = model(x, return_embedding=True)
            all_embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(all_embeddings, axis=0)

    emb_cols = [f"cnn_emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    out_df = pd.concat([df[["UID", "PHASE"]].reset_index(drop=True), emb_df], axis=1)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} row(s) x {len(emb_cols)} CNN feature(s) to {args.out}")


if __name__ == "__main__":
    main()
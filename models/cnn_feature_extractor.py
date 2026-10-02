"""
models/cnn_feature_extractor.py

Deep learning feature extractor that processes CT slices to learn
patient-specific anatomical patterns. This replaces manual feature engineering
with learned visual features from the actual medical images.

Architecture: ResNet-based encoder pretrained on ImageNet, fine-tuned on CT scans
Output: 128-dimensional embedding per patient capturing body habitus, tissue
        distribution, and scan-specific patterns
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os


class CTSliceDataset(Dataset):
    """Dataset for CT slice images"""

    def __init__(self, slice_paths, transform=None):
        self.slice_paths = slice_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel for pretrained models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        img = Image.open(self.slice_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img


class CTFeatureExtractor(nn.Module):
    """
    ResNet18-based feature extractor for CT scans.
    Outputs 128-dim embedding per patient.
    """

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Add projection head to get fixed-size embedding
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 3, 224, 224)
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        return embeddings

    def extract_patient_embedding(self, slice_paths, device='cpu', num_slices=10):
        """
        Extract single embedding for a patient by averaging across multiple slices.

        Args:
            slice_paths: List of paths to CT slice images for this patient
            device: 'cpu' or 'cuda'
            num_slices: Number of slices to sample (evenly spaced through scan)

        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        self.eval()
        self.to(device)

        # Sample evenly spaced slices
        if len(slice_paths) > num_slices:
            indices = np.linspace(0, len(slice_paths)-1, num_slices, dtype=int)
            slice_paths = [slice_paths[i] for i in indices]

        dataset = CTSliceDataset(slice_paths)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                emb = self.forward(batch)
                embeddings.append(emb.cpu().numpy())

        # Average across all slices
        all_embeddings = np.concatenate(embeddings, axis=0)
        patient_embedding = np.mean(all_embeddings, axis=0)

        return patient_embedding


def extract_features_from_dicom(dicom_dir, model, device='cpu', output_dir='_temp_slices'):
    """
    Extract CNN features from a DICOM folder.

    Args:
        dicom_dir: Path to folder containing DICOM files
        model: CTFeatureExtractor instance
        device: 'cpu' or 'cuda'
        output_dir: Temporary directory to save slice PNGs

    Returns:
        embedding: numpy array of shape (128,)
    """
    from src.dicom_io import load_dicom_series
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load DICOM and convert to HU
        volume_hu, datasets = load_dicom_series(dicom_dir)

        # Window for soft tissue visualization (standard abdominal CT window)
        # Window center: 40 HU, Window width: 400 HU
        lower = 40 - 400/2
        upper = 40 + 400/2
        volume_windowed = np.clip(volume_hu, lower, upper)
        volume_windowed = ((volume_windowed - lower) / (upper - lower) * 255).astype(np.uint8)

        # Save representative slices as PNG
        num_slices = volume_windowed.shape[2]
        slice_paths = []

        # Sample 10 evenly spaced slices
        indices = np.linspace(0, num_slices-1, min(10, num_slices), dtype=int)
        for i, idx in enumerate(indices):
            slice_img = Image.fromarray(volume_windowed[:, :, idx], mode='L')
            path = os.path.join(output_dir, f'slice_{i:03d}.png')
            slice_img.save(path)
            slice_paths.append(path)

        # Extract embedding
        embedding = model.extract_patient_embedding(slice_paths, device=device)

        return embedding

    finally:
        # Cleanup temp files
        shutil.rmtree(output_dir, ignore_errors=True)


def save_model(model, path):
    """Save model weights"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, embedding_dim=128, device='cpu'):
    """Load model weights"""
    model = CTFeatureExtractor(embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model

"""
Transformer-Based Organ Interaction Model

Uses attention mechanism to learn how organs within the same patient
influence each other's doses. Captures:
- Spatial relationships between organs
- Patient-specific shielding effects
- Scatter radiation patterns
- Organ proximity effects
"""
import torch
import torch.nn as nn
import math


class OrganAttentionModel(nn.Module):
    """
    Transformer that models organ interactions within a patient.

    Each patient scan becomes a sequence of organs, and we learn
    how each organ's dose is influenced by other organs in the same scan.
    """

    def __init__(self, n_organs=15, feature_dim=64, n_heads=4, n_layers=3):
        super().__init__()

        self.n_organs = n_organs
        self.feature_dim = feature_dim

        # Organ embedding
        self.organ_embedding = nn.Embedding(n_organs, feature_dim)

        # Feature projection for organ-level features
        self.feature_proj = nn.Linear(10, feature_dim)  # 10 input features

        # Positional encoding (spatial position in body)
        self.pos_encoding = nn.Parameter(torch.randn(15, feature_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim * 2,  # organ embedding + features
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, organ_indices, organ_features, mask=None):
        """
        Args:
            organ_indices: (batch, max_organs) - organ type indices
            organ_features: (batch, max_organs, 10) - per-organ features
            mask: (batch, max_organs) - True for padding positions

        Returns:
            doses: (batch, max_organs) - predicted doses for each organ
        """
        batch_size, max_organs = organ_indices.shape

        # Get organ embeddings
        organ_emb = self.organ_embedding(organ_indices)  # (batch, max_organs, feature_dim)

        # Project features
        feat_emb = self.feature_proj(organ_features)  # (batch, max_organs, feature_dim)

        # Add positional encoding
        pos_enc = self.pos_encoding[organ_indices]  # (batch, max_organs, feature_dim)

        # Combine embeddings
        combined = torch.cat([organ_emb + pos_enc, feat_emb], dim=-1)

        # Apply transformer
        if mask is not None:
            transformer_out = self.transformer(combined, src_key_padding_mask=mask)
        else:
            transformer_out = self.transformer(combined)

        # Predict doses
        doses = self.output(transformer_out).squeeze(-1)  # (batch, max_organs)

        return doses


class PatientGrouper:
    """
    Helper class to group organs by patient for batch processing.
    """

    @staticmethod
    def group_by_patient(df):
        """
        Group organ dataframe by (UID, PHASE) to process all organs together.

        Returns:
            patient_groups: List of DataFrames, one per patient
        """
        grouped = df.groupby(['UID', 'PHASE'])
        return [group for _, group in grouped]

    @staticmethod
    def collate_patient_batch(patient_dfs, max_organs=15):
        """
        Collate multiple patients into a batch tensor.

        Args:
            patient_dfs: List of DataFrames, one per patient
            max_organs: Maximum number of organs (for padding)

        Returns:
            organ_indices: (batch, max_organs) - padded organ indices
            organ_features: (batch, max_organs, 10) - padded features
            mask: (batch, max_organs) - True for padding positions
            lengths: (batch,) - actual number of organs per patient
        """
        batch_size = len(patient_dfs)

        organ_indices = torch.zeros(batch_size, max_organs, dtype=torch.long)
        organ_features = torch.zeros(batch_size, max_organs, 10)
        mask = torch.ones(batch_size, max_organs, dtype=torch.bool)
        lengths = []

        for i, df in enumerate(patient_dfs):
            n_organs = min(len(df), max_organs)
            lengths.append(n_organs)

            # Fill in actual data
            organ_indices[i, :n_organs] = torch.tensor(df['organ_encoded'].values[:n_organs])
            organ_features[i, :n_organs] = torch.tensor(df[feature_cols].values[:n_organs])
            mask[i, :n_organs] = False  # Not masked

        return organ_indices, organ_features, mask, lengths


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    max_organs = 15
    n_organs = 15

    model = OrganAttentionModel(n_organs=n_organs)

    # Dummy inputs
    organ_indices = torch.randint(0, n_organs, (batch_size, max_organs))
    organ_features = torch.randn(batch_size, max_organs, 10)
    mask = torch.zeros(batch_size, max_organs, dtype=torch.bool)
    mask[:, 10:] = True  # Mask last 5 organs (padding)

    # Forward pass
    doses = model(organ_indices, organ_features, mask)

    print(f"Input shape: {organ_features.shape}")
    print(f"Output shape: {doses.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nPredicted doses (first patient): {doses[0, :10]}")

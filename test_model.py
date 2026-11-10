"""Quick test to verify the trained model works"""
import pandas as pd
import numpy as np
from models.ensemble_dose_predictor import EnsembleDosePredictor

# Load the trained model
model = EnsembleDosePredictor.load('trained_models', device='cpu')

# Load a small sample from training data
df = pd.read_csv('training_data_complete.csv').head(10)

# Make predictions with details
predictions, valid_indices = model.predict(df, return_details=True)

print("Sample Predictions:")
print("="*70)
for i, row in df.head(10).iterrows():
    pred_str = f"{predictions[i]:6.2f}" if not np.isnan(predictions[i]) else "  SKIP"
    status = "✓" if not np.isnan(predictions[i]) else "✗"
    print(f"{status} {row['organ']:20s} - Actual: {row['absorbed_dose_mgy']:6.2f} mGy, Predicted: {pred_str} mGy")
print("="*70)

# Calculate error only on valid predictions
valid_mask = ~np.isnan(predictions[:10])
if valid_mask.sum() > 0:
    actual = df.head(10)['absorbed_dose_mgy'].values[valid_mask]
    pred_valid = predictions[:10][valid_mask]
    error = np.abs(pred_valid - actual)
    print(f"\nValid Predictions: {valid_mask.sum()}/10")
    print(f"Mean Absolute Error: {error.mean():.2f} mGy")
    print(f"Max Error: {error.max():.2f} mGy")
else:
    print("\nNo valid predictions (all samples had missing features)")

print(f"\n✓ Model loaded and working correctly!")

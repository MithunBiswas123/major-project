"""Quick diagnostic: compare old vs new sign data formats."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/sign_dataset.csv', low_memory=False)
feat_cols = [c for c in df.columns if c.startswith('feature_')]

print(f"Total samples: {len(df)}, Total signs: {df['sign'].nunique()}")
print(f"Sign list: {sorted(df['sign'].unique())}")
print(f"Feature columns: {len(feat_cols)}")
print()

# Per-sign sample counts and z-weight analysis
counts = df['sign'].value_counts().sort_index()
for sign, cnt in counts.items():
    feats = df[df['sign']==sign][feat_cols].values.astype(float)
    feats = np.nan_to_num(feats, nan=0.0)
    
    left = feats[:, :63]
    right = feats[:, 63:126]
    
    left_active = (left != 0).any(axis=1).mean() * 100
    right_active = (right != 0).any(axis=1).mean() * 100
    
    # Check z-weight
    z_weights = []
    for row in feats[:20]:
        for start in [0, 63]:
            hand = row[start:start+63]
            if np.any(hand != 0):
                lm = hand.reshape(21, 3)
                wrist = lm[0]
                rel = lm[1:] - wrist
                xy_norm = np.linalg.norm(rel[:, :2])
                z_norm = np.linalg.norm(rel[:, 2])
                total = np.linalg.norm(rel)
                if total > 1e-8:
                    z_weights.append(z_norm / total)
    
    z_w = np.mean(z_weights) if z_weights else 0
    print(f"{sign:12s}: {cnt:5d} samples, L={left_active:5.1f}% R={right_active:5.1f}%, z_weight={z_w:.4f}")

# Show raw feature ranges for a few signs
print("\n\n=== RAW FEATURE RANGES ===")
for sign in ['A', 'B', 'hello', 'yes', 'hi', 'is', 'name', 'what']:
    subset = df[df['sign'] == sign]
    if len(subset) == 0:
        continue
    feats = subset[feat_cols].values.astype(float)
    feats = np.nan_to_num(feats, nan=0.0)
    nz = feats[feats != 0]
    if len(nz) > 0:
        print(f"  {sign:10s}: range=[{nz.min():.4f}, {nz.max():.4f}], mean={nz.mean():.4f}")
    
    # Show first sample wrist coords for right hand
    row = feats[0]
    rh = row[63:126].reshape(21, 3)
    lh = row[:63].reshape(21, 3)
    if np.any(rh != 0):
        print(f"             RH wrist={rh[0]}, fingertip5={rh[8]}")
    if np.any(lh != 0):
        print(f"             LH wrist={lh[0]}, fingertip5={lh[8]}")

# Now check the label encoder
import joblib
le = joblib.load('models/saved/label_encoder.pkl')
print(f"\n\n=== LABEL ENCODER ===")
print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

# Check which signs from the dataset are in the label encoder
dataset_signs = set(df['sign'].unique())
encoder_signs = set(le.classes_)
print(f"\nIn dataset but NOT in encoder: {dataset_signs - encoder_signs}")
print(f"In encoder but NOT in dataset: {encoder_signs - dataset_signs}")

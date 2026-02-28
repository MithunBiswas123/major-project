"""Check if old data is synthetic (random) or real webcam data."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/sign_dataset.csv', low_memory=False)
feat_cols = [c for c in df.columns if c.startswith('feature_')]

print("=== CHECKING IF OLD DATA IS SYNTHETIC ===\n")

# 1. Check correlation between samples of the same sign
# Synthetic data: all samples share same base_pattern + noise, so very correlated
# Real data: more variation depending on hand position in frame
for sign in ['A', 'B', 'hello', 'hi']:
    subset = df[df['sign'] == sign][feat_cols].values.astype(float)
    subset = np.nan_to_num(subset, nan=0.0)
    
    # Compute correlation between first 5 samples
    if len(subset) >= 5:
        corrs = []
        for i in range(5):
            for j in range(i+1, 5):
                c = np.corrcoef(subset[i], subset[j])[0, 1]
                corrs.append(c)
        print(f"  {sign:10s}: avg correlation between samples = {np.mean(corrs):.4f} "
              f"(synthetic data would be ~0.96+, real data ~0.5-0.9)")

# 2. Check if landmark 9 (middle finger MCP) has consistent distance from wrist
print("\n=== LANDMARK 9 ANALYSIS (middle finger MCP) ===")
for sign in ['A', 'hello', '1', 'hi', 'is']:
    subset = df[df['sign'] == sign][feat_cols].values.astype(float)
    subset = np.nan_to_num(subset, nan=0.0)
    
    dists = []
    for row in subset:
        # Find which hand has data
        left = row[:63].reshape(21, 3)
        right = row[63:].reshape(21, 3)
        
        for hand in [left, right]:
            if np.any(hand != 0):
                # landmark 9 distance from landmark 0
                d = np.linalg.norm(hand[9] - hand[0])
                if d > 0:
                    dists.append(d)
    
    if dists:
        print(f"  {sign:10s}: lm9 dist mean={np.mean(dists):.4f}, "
              f"std={np.std(dists):.4f}, range=[{np.min(dists):.4f}, {np.max(dists):.4f}]")

# 3. Check consistency WITHIN a sign's samples
print("\n=== FEATURE VALUE DISTRIBUTION (per-sign std of each feature) ===")
for sign in ['A', 'hello', 'hi']:
    subset = df[df['sign'] == sign][feat_cols].values.astype(float)
    subset = np.nan_to_num(subset, nan=0.0)
    
    # Only non-zero columns
    nonzero_mask = (subset != 0).any(axis=0)
    if nonzero_mask.any():
        per_feat_std = subset[:, nonzero_mask].std(axis=0)
        print(f"  {sign:10s}: avg per-feature std = {per_feat_std.mean():.4f} "
              f"(synthetic=~0.1, real=much higher)")

# 4. Direct comparison: normalize old feature by landmark 9
print("\n=== AFTER LANDMARK-9 NORMALIZATION ===")
for sign in ['A', 'hello', 'hi', 'is']:
    subset = df[df['sign'] == sign][feat_cols].values.astype(float)
    subset = np.nan_to_num(subset, nan=0.0)
    
    normalized = []
    for row in subset:
        left = row[:63].reshape(21, 3)
        right = row[63:].reshape(21, 3)
        
        for hand in [left, right]:
            if np.any(hand != 0):
                wrist = hand[0].copy()
                hand_rel = hand - wrist
                d = np.linalg.norm(hand_rel[9])
                if d > 1e-6:
                    hand_norm = hand_rel / d
                    normalized.append(hand_norm.flatten())
    
    if normalized:
        all_norm = np.array(normalized)
        print(f"  {sign:10s}: after norm range=[{all_norm.min():.3f}, {all_norm.max():.3f}], "
              f"mean={all_norm.mean():.4f}, std={all_norm.std():.4f}")

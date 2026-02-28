"""Deep diagnostic: compare feature distributions between old & new signs,
and check which hand slots have data."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/sign_dataset.csv', low_memory=False)
feat_cols = [c for c in df.columns if c.startswith('feature_')]

old_signs = ['A', 'B', 'C', 'D', 'E', '1', '2', '3', '4', '5',
             'hello', 'yes', 'no', 'stop', 'eat', 'drink']
new_signs = ['hi', 'how', 'is', 'your', 'name', 'my', 'to', 'who',
             'where', 'why', 'when', 'which', 'me', 'what']

print("=" * 70)
print("FEATURE SLOT ANALYSIS: Which hand slot has data?")
print("  Features 0-62  = LEFT hand")
print("  Features 63-125 = RIGHT hand")
print("=" * 70)

for label, signs in [("OLD SIGNS", old_signs), ("NEW SIGNS", new_signs)]:
    print(f"\n--- {label} ---")
    for sign in signs:
        subset = df[df['sign'] == sign]
        if len(subset) == 0:
            continue
        feats = subset[feat_cols].values.astype(float)
        feats = np.nan_to_num(feats, nan=0.0)
        
        left = feats[:, :63]   # features 0-62
        right = feats[:, 63:]  # features 63-125
        
        left_nz = (left != 0).any(axis=1).mean() * 100   # % of samples with left hand
        right_nz = (right != 0).any(axis=1).mean() * 100  # % of samples with right hand
        
        left_mean = left[left != 0].mean() if (left != 0).any() else 0
        right_mean = right[right != 0].mean() if (right != 0).any() else 0
        
        left_range = f"[{left.min():.3f}, {left.max():.3f}]" if (left != 0).any() else "N/A"
        right_range = f"[{right.min():.3f}, {right.max():.3f}]" if (right != 0).any() else "N/A"
        
        print(f"  {sign:10s}: LEFT={left_nz:5.1f}% range={left_range:30s}  "
              f"RIGHT={right_nz:5.1f}% range={right_range}")

print("\n" + "=" * 70)
print("SAMPLE FEATURE VALUES (first sample of each)")
print("=" * 70)
for sign in ['A', 'hi']:
    row = df[df['sign'] == sign].iloc[0]
    feats = row[feat_cols].values.astype(float)
    feats = np.nan_to_num(feats, nan=0.0)
    
    print(f"\n  Sign '{sign}':")
    print(f"    Left[0:6]  (x,y,z of landmark 0 and 1): {feats[0:6]}")
    print(f"    Left[60:63] (x,y,z of landmark 20):     {feats[60:63]}")
    print(f"    Right[0:6] (x,y,z of landmark 0 and 1): {feats[63:69]}")
    print(f"    Right[60:63](x,y,z of landmark 20):     {feats[123:126]}")

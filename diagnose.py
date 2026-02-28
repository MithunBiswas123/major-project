"""Quick diagnostic: does the saved model work on raw CSV data?"""
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

model = tf.keras.models.load_model('models/saved/hybrid_model.h5')
le = joblib.load('models/saved/label_encoder.pkl')
scaler = joblib.load('models/saved/scaler.pkl')

df = pd.read_csv('data/raw/sign_dataset.csv', low_memory=False)
feat_cols = [c for c in df.columns if c.startswith('feature_')]

print(f"Scaler mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
print(f"Scaler scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
print(f"Scaler n_features: {scaler.n_features_in_}")
print()

# Check some raw feature stats
for sign in ['A', 'hi', 'hello', 'yes', '1']:
    subset = df[df['sign'] == sign]
    feats = subset[feat_cols].values.astype(float)
    feats = np.nan_to_num(feats, nan=0.0)
    non_zero_pct = (feats != 0).mean() * 100
    print(f"Sign '{sign}': {len(subset)} samples, non-zero={non_zero_pct:.1f}%, "
          f"mean={feats.mean():.4f}, std={feats.std():.4f}, "
          f"range=[{feats.min():.4f}, {feats.max():.4f}]")

print("\n--- Running predictions (same path as detect_sign.py) ---\n")

correct = 0
total = 0
per_sign = {}

for sign in sorted(df['sign'].unique()):
    subset = df[df['sign'] == sign].sample(min(5, len(df[df['sign'] == sign])), random_state=42)
    sign_correct = 0
    sign_total = 0
    confs = []
    
    for _, row in subset.iterrows():
        feats = row[feat_cols].values.astype(float)
        feats = np.nan_to_num(feats, nan=0.0)
        feats_scaled = scaler.transform(feats.reshape(1, -1))
        pred = model.predict(feats_scaled, verbose=0)
        pred_idx = np.argmax(pred[0])
        pred_sign = le.classes_[pred_idx]
        conf = float(pred[0][pred_idx])
        
        is_correct = (pred_sign == sign)
        correct += int(is_correct)
        total += 1
        sign_correct += int(is_correct)
        sign_total += 1
        confs.append(conf)
    
    avg_conf = np.mean(confs)
    status = "OK" if sign_correct == sign_total else "FAIL"
    print(f"  {sign:12s}: {sign_correct}/{sign_total}  avg_conf={avg_conf:.3f}  [{status}]")

print(f"\nOverall: {correct}/{total} = {correct/total*100:.1f}%")

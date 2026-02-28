"""
Diagnose WHY old signs fail at webcam inference but new signs work.

Key hypothesis: old data was collected in a different coordinate system 
(possibly pixel-scaled), so after L2 normalization the z-coordinate has 
different relative weight → features don't match webcam output.
"""
import numpy as np
import pandas as pd

CSV = "data/raw/sign_dataset.csv"
df = pd.read_csv(CSV)
feat_cols = [c for c in df.columns if c.startswith('feature_')]

# ── pick one old sign sample and one new sign sample ──
old_signs = ['A', 'B', 'hello', 'help', 'yes', 'no', 'C', 'happy']
new_signs = ['hi', 'how', 'is', 'my', 'name', 'what', 'where']

def analyze_hand(hand_63, label):
    """Analyze one hand slot (63 features)."""
    if np.all(hand_63 == 0):
        return None
    lm = hand_63.reshape(21, 3)
    wrist = lm[0]
    x_vals = lm[:, 0]
    y_vals = lm[:, 1]
    z_vals = lm[:, 2]
    
    # Wrist-subtracted
    lm_ws = lm - wrist
    x_rel = lm_ws[1:, 0]  # non-wrist
    y_rel = lm_ws[1:, 1]
    z_rel = lm_ws[1:, 2]
    
    xy_norm = np.linalg.norm(np.column_stack([x_rel, y_rel]))
    z_norm = np.linalg.norm(z_rel)
    total_norm = np.linalg.norm(lm_ws[1:])
    
    z_relative_weight = z_norm / total_norm if total_norm > 1e-8 else 0
    
    return {
        'wrist': wrist,
        'x_range': (x_vals.min(), x_vals.max()),
        'y_range': (y_vals.min(), y_vals.max()),
        'z_range': (z_vals.min(), z_vals.max()),
        'x_abs': np.max(np.abs(x_rel)),
        'y_abs': np.max(np.abs(y_rel)),
        'z_abs': np.max(np.abs(z_rel)),
        'xy_norm': xy_norm,
        'z_norm': z_norm,
        'total_norm': total_norm,
        'z_relative_weight': z_relative_weight,
    }

print("=" * 80)
print("COMPARING OLD vs NEW DATA: z-coordinate relative weight")
print("=" * 80)

for group, signs in [("OLD", old_signs), ("NEW", new_signs)]:
    print(f"\n{'='*40} {group} SIGNS {'='*40}")
    for sign in signs:
        samples = df[df['sign'] == sign]
        if len(samples) == 0:
            continue
        
        z_weights_right = []
        z_weights_left = []
        z_abs_right = []
        z_abs_left = []
        wrist_mags = []
        
        for _, row in samples.head(50).iterrows():
            feats = row[feat_cols].values.astype(float)
            
            # Left hand [0:63]
            info_l = analyze_hand(feats[:63], f"{sign}_L")
            if info_l:
                z_weights_left.append(info_l['z_relative_weight'])
                z_abs_left.append(info_l['z_abs'])
                wrist_mags.append(np.linalg.norm(info_l['wrist']))
            
            # Right hand [63:126]
            info_r = analyze_hand(feats[63:126], f"{sign}_R")
            if info_r:
                z_weights_right.append(info_r['z_relative_weight'])
                z_abs_right.append(info_r['z_abs'])
                wrist_mags.append(np.linalg.norm(info_r['wrist']))
        
        z_w = z_weights_right if z_weights_right else z_weights_left
        z_a = z_abs_right if z_abs_right else z_abs_left
        if z_w:
            print(f"  {sign:12s}: z_weight={np.mean(z_w):.4f} ± {np.std(z_w):.4f}  "
                  f"z_abs={np.mean(z_a):.4f}  wrist_mag={np.mean(wrist_mags):.4f}")

# ── Now test what L2-normalized features look like ──
print("\n\n" + "=" * 80)
print("AFTER L2 NORMALIZATION: feature comparison")
print("=" * 80)

from src.preprocessing import normalize_hand_features

def compare_normalized(sign, n=20):
    """Show normalized feature stats for a sign."""
    samples = df[df['sign'] == sign].head(n)
    if len(samples) == 0:
        return
    
    normed_all = []
    for _, row in samples.iterrows():
        feats = row[feat_cols].values.astype(float)
        normed = normalize_hand_features(feats)
        normed_all.append(normed)
    
    normed_all = np.array(normed_all)
    
    # Analyze right hand (features 63:126)
    rh = normed_all[:, 63:126]
    if np.any(rh != 0):
        rh_lm = rh.reshape(-1, 21, 3)
        # non-wrist landmarks
        nw = rh_lm[:, 1:, :]
        print(f"  {sign:12s} (R): x=[{nw[:,:,0].mean():.4f}±{nw[:,:,0].std():.4f}]  "
              f"y=[{nw[:,:,1].mean():.4f}±{nw[:,:,1].std():.4f}]  "
              f"z=[{nw[:,:,2].mean():.4f}±{nw[:,:,2].std():.4f}]  "
              f"z/xy_ratio={np.std(nw[:,:,2]) / (np.std(nw[:,:,:2]) + 1e-8):.4f}")
    
    lh = normed_all[:, :63]
    if np.any(lh != 0):
        lh_lm = lh.reshape(-1, 21, 3)
        nw = lh_lm[:, 1:, :]
        print(f"  {sign:12s} (L): x=[{nw[:,:,0].mean():.4f}±{nw[:,:,0].std():.4f}]  "
              f"y=[{nw[:,:,1].mean():.4f}±{nw[:,:,1].std():.4f}]  "
              f"z=[{nw[:,:,2].mean():.4f}±{nw[:,:,2].std():.4f}]  "
              f"z/xy_ratio={np.std(nw[:,:,2]) / (np.std(nw[:,:,:2]) + 1e-8):.4f}")

for group, signs in [("OLD", old_signs), ("NEW", new_signs)]:
    print(f"\n--- {group} SIGNS after normalize_hand_features ---")
    for sign in signs:
        compare_normalized(sign)

# ── KEY TEST: What if we normalize xy and z separately? ──
print("\n\n" + "=" * 80)
print("ALTERNATIVE: Separate xy/z normalization")
print("=" * 80)

def normalize_hand_separate_xyzn(features_126):
    """Wrist-subtract, normalize xy together and z separately."""
    result = features_126.copy().astype(np.float64)
    
    for start in [0, 63]:
        hand = result[start:start+63].copy()
        if np.any(hand != 0):
            landmarks = hand.reshape(21, 3)
            wrist = landmarks[0].copy()
            landmarks = landmarks - wrist
            
            non_wrist = landmarks[1:]  # 20 landmarks
            
            # Normalize xy together
            xy = non_wrist[:, :2]
            xy_norm = np.linalg.norm(xy)
            if xy_norm > 1e-6:
                non_wrist[:, :2] = xy / xy_norm
            
            # Normalize z separately
            zvals = non_wrist[:, 2]
            z_norm = np.linalg.norm(zvals)
            if z_norm > 1e-6:
                non_wrist[:, 2] = zvals / z_norm
            
            landmarks[1:] = non_wrist
            result[start:start+63] = landmarks.flatten()
    
    return result

def compare_alt_normalized(sign, n=20):
    samples = df[df['sign'] == sign].head(n)
    if len(samples) == 0:
        return
    
    normed_all = []
    for _, row in samples.iterrows():
        feats = row[feat_cols].values.astype(float)
        normed = normalize_hand_separate_xyzn(feats)
        normed_all.append(normed)
    
    normed_all = np.array(normed_all)
    
    rh = normed_all[:, 63:126]
    if np.any(rh != 0):
        rh_lm = rh.reshape(-1, 21, 3)
        nw = rh_lm[:, 1:, :]
        print(f"  {sign:12s} (R): x=[{nw[:,:,0].mean():.4f}±{nw[:,:,0].std():.4f}]  "
              f"y=[{nw[:,:,1].mean():.4f}±{nw[:,:,1].std():.4f}]  "
              f"z=[{nw[:,:,2].mean():.4f}±{nw[:,:,2].std():.4f}]")
    
    lh = normed_all[:, :63]
    if np.any(lh != 0):
        lh_lm = lh.reshape(-1, 21, 3)
        nw = lh_lm[:, 1:, :]
        print(f"  {sign:12s} (L): x=[{nw[:,:,0].mean():.4f}±{nw[:,:,0].std():.4f}]  "
              f"y=[{nw[:,:,1].mean():.4f}±{nw[:,:,1].std():.4f}]  "
              f"z=[{nw[:,:,2].mean():.4f}±{nw[:,:,2].std():.4f}]")

for group, signs in [("OLD", old_signs), ("NEW", new_signs)]:
    print(f"\n--- {group} SIGNS ---")
    for sign in signs:
        compare_alt_normalized(sign)

# ── SIMULATE WEBCAM: take new sign raw format and compare with old sign ──
print("\n\n" + "=" * 80)
print("CRITICAL TEST: Cosine similarity between old-format and new-format")
print("after normalization for the SAME hand shape (approximation)")
print("=" * 80)

# For signs that exist in both old and new, we can't compare directly
# But we can check: if old data is rescaled to [0,1] range first,
# then normalized, does it produce similar features?

def convert_old_to_raw_like(features_126):
    """Convert old wrist-normalized data to raw-MediaPipe-like format."""
    result = features_126.copy().astype(np.float64)
    
    for start in [0, 63]:
        hand = result[start:start+63].copy()
        if np.any(hand != 0):
            landmarks = hand.reshape(21, 3)
            wrist_mag = np.linalg.norm(landmarks[0])
            max_abs = np.max(np.abs(landmarks))
            
            if wrist_mag < 0.1 and max_abs > 1.5:
                # Scale to [-0.5, 0.5] then shift to center at 0.5
                landmarks = landmarks / (2 * max_abs)
                landmarks = landmarks + np.array([0.5, 0.5, 0.0])
            
            result[start:start+63] = landmarks.flatten()
    
    return result

# Test: old data → convert_to_raw → normalize  vs  old data → normalize directly
print("\nComparing: direct normalize vs convert-to-raw-then-normalize (for old signs)")
for sign in ['A', 'hello', 'help', 'yes']:
    samples = df[df['sign'] == sign].head(10)
    if len(samples) == 0:
        continue
    
    cosines = []
    for _, row in samples.iterrows():
        feats = row[feat_cols].values.astype(float)
        
        # Direct normalize (what training does)
        direct = normalize_hand_features(feats)
        
        # Convert to raw, then normalize (what webcam would produce, approximately)
        raw_like = convert_old_to_raw_like(feats)
        via_raw = normalize_hand_features(raw_like)
        
        # Cosine similarity
        d_norm = np.linalg.norm(direct)
        v_norm = np.linalg.norm(via_raw)
        if d_norm > 1e-8 and v_norm > 1e-8:
            cos = np.dot(direct, via_raw) / (d_norm * v_norm)
            cosines.append(cos)
    
    if cosines:
        print(f"  {sign:12s}: cosine_sim = {np.mean(cosines):.4f} ± {np.std(cosines):.4f}")
        # If cosine < 0.99, the features are significantly different

# ── Test what accuracy we get with convert_old_to_raw_like + normalize ──
print("\n\n" + "=" * 80)
print("QUICK MODEL TEST: convert_old_to_raw + normalize vs direct normalize")
print("=" * 80)

import tensorflow as tf
import joblib

model = tf.keras.models.load_model('models/saved/hybrid_model.h5')
le = joblib.load('models/saved/label_encoder.pkl')
scaler = joblib.load('models/saved/scaler.pkl')

# Test with current pipeline (direct normalize - as model was trained)
correct_direct = 0
correct_via_raw = 0
total = 0

old_correct_direct = 0
old_correct_via_raw = 0
old_total = 0

new_correct_direct = 0
new_correct_via_raw = 0
new_total = 0

old_set = set(['1','2','3','4','5','A','B','C','D','E','drink','eat','happy','hello',
               'help','no','peace','run','sick','sorry','stop','thankyou','thirsty',
               'tired','wait','welcome','yes'])
new_set = set(['hi','how','is','me','my','name','to','what','when','where','which','who','why','your'])

for sign in df['sign'].unique():
    samp = df[df['sign'] == sign].sample(min(20, len(df[df['sign']==sign])), random_state=42)
    
    for _, row in samp.iterrows():
        feats = row[feat_cols].values.astype(float)
        true_label = row['sign']
        total += 1
        is_old = true_label in old_set
        if is_old:
            old_total += 1
        else:
            new_total += 1
        
        # Method 1: Direct normalize (what model was trained with)
        normed = normalize_hand_features(feats)
        scaled = scaler.transform(normed.reshape(1, -1))
        pred = model.predict(scaled, verbose=0)
        pred_label = le.classes_[np.argmax(pred)]
        if pred_label == true_label:
            correct_direct += 1
            if is_old: old_correct_direct += 1
            else: new_correct_direct += 1
        
        # Method 2: Convert to raw first, then normalize (simulating webcam)
        raw_like = convert_old_to_raw_like(feats)
        normed2 = normalize_hand_features(raw_like)
        scaled2 = scaler.transform(normed2.reshape(1, -1))
        pred2 = model.predict(scaled2, verbose=0)
        pred_label2 = le.classes_[np.argmax(pred2)]
        if pred_label2 == true_label:
            correct_via_raw += 1
            if is_old: old_correct_via_raw += 1
            else: new_correct_via_raw += 1

print(f"\nDirect normalize (train pipeline):    {correct_direct}/{total} = {correct_direct/total*100:.1f}%")
print(f"  Old signs: {old_correct_direct}/{old_total} = {old_correct_direct/old_total*100:.1f}%")
print(f"  New signs: {new_correct_direct}/{new_total} = {new_correct_direct/new_total*100:.1f}%")

print(f"\nConvert-to-raw + normalize (webcam sim): {correct_via_raw}/{total} = {correct_via_raw/total*100:.1f}%")
print(f"  Old signs: {old_correct_via_raw}/{old_total} = {old_correct_via_raw/old_total*100:.1f}%")
print(f"  New signs: {new_correct_via_raw}/{new_total} = {new_correct_via_raw/new_total*100:.1f}%")

print("\n\nIf 'convert-to-raw' accuracy drops for old signs, it proves the")
print("model learned old-format-specific features that webcam can't reproduce.")
print("The fix would be to train with convert_old_to_raw applied BEFORE normalize.")

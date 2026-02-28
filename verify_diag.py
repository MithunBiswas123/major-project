"""Verify: model accuracy on old signs vs new signs from CSV data."""
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import normalize_hand_features

model = tf.keras.models.load_model('models/saved/hybrid_model.h5')
le = joblib.load('models/saved/label_encoder.pkl')
scaler = joblib.load('models/saved/scaler.pkl')

df = pd.read_csv('data/raw/sign_dataset.csv', low_memory=False)
feat_cols = [c for c in df.columns if c.startswith('feature_')]

old_signs = {'1','2','3','4','5','A','B','C','D','E','drink','eat','happy','hello',
             'help','no','peace','run','sick','sorry','stop','thankyou','thirsty',
             'tired','wait','welcome','yes'}
new_signs = {'hi','how','is','me','my','name','to','what','when','where','which','who','why','your'}

print("=== MODEL ACCURACY ON CSV DATA (with normalize_hand_features) ===\n")

old_correct = 0; old_total = 0
new_correct = 0; new_total = 0

for sign in sorted(df['sign'].unique()):
    subset = df[df['sign'] == sign].sample(min(20, len(df[df['sign']==sign])), random_state=42)
    sign_ok = 0
    sign_n = 0
    
    for _, row in subset.iterrows():
        feats = row[feat_cols].values.astype(float)
        feats = np.nan_to_num(feats, nan=0.0)
        normed = normalize_hand_features(feats)
        scaled = scaler.transform(normed.reshape(1, -1))
        pred = model.predict(scaled, verbose=0)
        pred_label = le.classes_[np.argmax(pred[0])]
        conf = pred[0][np.argmax(pred[0])]
        
        if pred_label == sign:
            sign_ok += 1
            if sign in old_signs: old_correct += 1
            else: new_correct += 1
        
        sign_n += 1
        if sign in old_signs: old_total += 1
        else: new_total += 1
    
    group = "OLD" if sign in old_signs else "NEW"
    status = "OK" if sign_ok == sign_n else "FAIL"
    print(f"  [{group}] {sign:12s}: {sign_ok}/{sign_n} ({sign_ok/sign_n*100:.0f}%) [{status}]")

print(f"\n  OLD signs accuracy: {old_correct}/{old_total} = {old_correct/old_total*100:.1f}%")
print(f"  NEW signs accuracy: {new_correct}/{new_total} = {new_correct/new_total*100:.1f}%")
print(f"  Overall:            {old_correct+new_correct}/{old_total+new_total} = {(old_correct+new_correct)/(old_total+new_total)*100:.1f}%")

# Now simulate webcam: take a new sign raw sample and show what normalize does
print("\n\n=== SIMULATED WEBCAM: normalize a real webcam sample ===")
print("(What your webcam produces for 'hi' sign)")
hi_row = df[df['sign'] == 'hi'].iloc[0]
hi_feats = hi_row[feat_cols].values.astype(float)
hi_normed = normalize_hand_features(hi_feats)
hi_scaled = scaler.transform(hi_normed.reshape(1, -1))
pred = model.predict(hi_scaled, verbose=0)
top3 = np.argsort(pred[0])[-3:][::-1]
print(f"  hi sample: top3 = {[(le.classes_[i], f'{pred[0][i]*100:.1f}%') for i in top3]}")

print("\n(What your webcam would produce for 'A' sign - but old CSV data is NOT webcam format)")
a_row = df[df['sign'] == 'A'].iloc[0]
a_feats = a_row[feat_cols].values.astype(float)
a_normed = normalize_hand_features(a_feats)
a_scaled = scaler.transform(a_normed.reshape(1, -1))
pred = model.predict(a_scaled, verbose=0)
top3 = np.argsort(pred[0])[-3:][::-1]
print(f"  A sample (CSV): top3 = {[(le.classes_[i], f'{pred[0][i]*100:.1f}%') for i in top3]}")

# KEY: show range differences
print("\n\n=== THE ROOT CAUSE ===")
print("Old data (e.g. 'A'):")
print(f"  Raw range: [{a_feats[a_feats!=0].min():.2f}, {a_feats[a_feats!=0].max():.2f}]")
print(f"  Wrist at: {a_feats[:63].reshape(21,3)[0]}")
print("New data (e.g. 'hi'):")
print(f"  Raw range: [{hi_feats[hi_feats!=0].min():.4f}, {hi_feats[hi_feats!=0].max():.4f}]")
print(f"  Wrist at: {hi_feats[63:126].reshape(21,3)[0]}")
print("\nOld data is SYNTHETIC (pre-processed, wrist=[0,0,0], range [-30,16])")
print("New data is REAL WEBCAM (MediaPipe coords, wrist~[0.5,0.7,0], range [0,1])")
print("=> The model learned old-format patterns that webcam can NEVER reproduce!")

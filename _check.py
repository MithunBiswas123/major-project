import pandas as pd
df = pd.read_csv('data/raw/sign_dataset.csv', low_memory=False)
print(f"Columns: {list(df.columns[:5])}")
print(f"Total: {len(df)} samples, {df['sign'].nunique()} signs")
counts = df['sign'].value_counts().sort_index()
for s, c in counts.items():
    print(f"  {s:12s}: {c}")

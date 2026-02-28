"""Find the full extract_landmarks and synthetic data code from notebook."""
import json

with open('notebooks/Complete_Sign_Language_ML.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print cells 7 and 8 completely
for i in [7, 8]:
    cell = nb['cells'][i]
    src = ''.join(cell['source'])
    print(f"=== Cell {i} ({cell['cell_type']}) ===")
    print(src)
    print("\n" + "=" * 80 + "\n")

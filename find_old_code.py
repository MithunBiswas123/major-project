"""Find the original feature extraction code from the notebook."""
import json

with open('notebooks/Complete_Sign_Language_ML.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

keywords = ['wrist', 'landmark', 'extract', 'lm.x', 'hand_size', 'normaliz', 
            'hand_span', 'subtract', 'relative']

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(kw in src.lower() for kw in keywords):
        ct = cell['cell_type']
        preview = src[:600]
        print(f"=== Cell {i} ({ct}) ===")
        print(preview)
        if len(src) > 600:
            print("... (truncated)")
        print()

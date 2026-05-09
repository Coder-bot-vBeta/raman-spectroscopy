import sys, numpy as np
sys.path.insert(0, '.')
from database.reference_store import get_database, invalidate_cache
from core.preprocessing import load_csv, preprocess
from core.spectral_matching import match

invalidate_cache()
matrix, names, metadata = get_database()
print(f"Reference DB: {len(names)} minerals, matrix {matrix.shape}")

# Check for bad reference spectra (zero or near-zero norms)
norms = np.linalg.norm(matrix, axis=1)
bad   = np.where(norms < 0.5)[0]
print(f"Bad references (norm < 0.5): {len(bad)}")
if len(bad):
    for i in bad[:5]:
        print(f"  {names[i]}: norm={norms[i]:.4f}, source={metadata[names[i]]['source']}")

# Check norm distribution
print(f"Norm stats: min={norms.min():.4f}  mean={norms.mean():.4f}  max={norms.max():.4f}")

# Test matching with a synthetic quartz spectrum
from database.synthetic_references import build_synthetic_db
import config
synth = build_synthetic_db()
q_synth = synth['Quartz']['spectrum']
results = match(q_synth)
print(f"\nSynthetic Quartz query -> top 5 matches:")
for r in results:
    print(f"  {r['mineral']:25s}  {r['confidence_pct']:5.1f}%  ({r['source']})")

# Test with a sample CSV file
import os
sample = 'samples/quartz.csv'
if os.path.exists(sample):
    with open(sample, 'rb') as f:
        raw = f.read()
    wn, inten = load_csv(raw)
    grid, proc, raw_g, steps = preprocess(wn, inten)
    q_real = np.array(proc, dtype=np.float32)
    results2 = match(q_real)
    print(f"\nSample quartz.csv query -> top 5 matches:")
    for r in results2:
        print(f"  {r['mineral']:25s}  {r['confidence_pct']:5.1f}%  ({r['source']})")

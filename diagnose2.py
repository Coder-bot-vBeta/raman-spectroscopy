import sys, numpy as np, glob, os
sys.path.insert(0, '.')
import config
from database.reference_store import invalidate_cache, get_database
from core.spectral_matching import _centre_normalise

invalidate_cache()
matrix, names, metadata = get_database()

# How many RRUFF files per mineral for suspects?
suspects = ['Quartz', 'Sodalite', 'Trattnerite', 'Amicite']
for mineral in suspects:
    files = glob.glob(os.path.join(config.PROCESSED_DIR, f"{mineral.replace(' ','_')}_*.npz"))
    print(f"{mineral}: {len(files)} RRUFF files")

print()
grid = config.GRID

# Find peak positions for each suspect
for mineral in suspects:
    if mineral not in names:
        print(f"{mineral}: NOT in DB"); continue
    idx  = names.index(mineral)
    spec = matrix[idx]  # already mean-centred + normalised
    # un-centre to find actual peak positions
    raw_files = glob.glob(os.path.join(config.PROCESSED_DIR, f"{mineral.replace(' ','_')}_*.npz"))
    if raw_files:
        raws = [np.load(f)['spectrum'] for f in raw_files]
        avg  = np.mean(raws, axis=0)
        top5_idx = np.argsort(avg)[::-1][:5]
        top5_wn  = grid[top5_idx]
        top5_v   = avg[top5_idx]
        print(f"{mineral} top peaks: {[f'{w:.0f}cm-1({v:.3f})' for w,v in zip(top5_wn, top5_v)]}")

print()
# Compute Pearson correlation of Quartz reference with Sodalite reference
q_idx = names.index('Quartz') if 'Quartz' in names else -1
s_idx = names.index('Sodalite') if 'Sodalite' in names else -1
if q_idx >= 0 and s_idx >= 0:
    corr = float(matrix[q_idx] @ matrix[s_idx])
    print(f"Quartz-Sodalite reference correlation: {corr:.4f}")
    corr2 = float(matrix[q_idx] @ matrix[q_idx])
    print(f"Quartz-Quartz self-correlation:        {corr2:.4f}")

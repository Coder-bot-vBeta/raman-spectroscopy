"""
Combined-metric spectral matching.

We score every reference row against the query with two complementary metrics:

  • L2 cosine on the L2-normalised intensity spectrum — sensitive to the full
    peak shape (positions, relative heights, breadth).
  • Pearson on the Savitzky–Golay 2nd derivative — emphasises narrow Raman
    peaks over broad fluorescence backgrounds and decorrelates near-DC drift.

The reference DB carries multiple rows per mineral (synthetic + RRUFF) so a
clean Lorentzian query and a noisy real-world query both have an ideal anchor.
We then dedupe the result list by mineral display-name, keeping each mineral's
best-scoring row.
"""
import numpy as np
from scipy.signal import savgol_filter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database.reference_store import get_match_database, get_database

_D2_WINDOW    = 15
_D2_POLYORDER = 3
_W_COSINE     = 0.5
_W_D2PEARSON  = 0.5


def _d2_pearson(v: np.ndarray) -> np.ndarray:
    v    = v.astype(np.float64)
    d2   = savgol_filter(v, window_length=_D2_WINDOW,
                         polyorder=_D2_POLYORDER, deriv=2)
    d2  -= d2.mean()
    norm = np.linalg.norm(d2)
    return (d2 / norm).astype(np.float32) if norm > 1e-12 else d2.astype(np.float32)


def match(query: np.ndarray, top_k: int = None) -> list:
    """
    Match query against the combined reference DB and return the top_k
    minerals (deduped by display name) sorted by combined similarity.
    """
    top_k = top_k or config.TOP_K

    d2_matrix, names, metadata = get_match_database()
    l2_matrix, l2_names, _      = get_database()
    if names != l2_names:
        # Defensive — both functions go through the same row builder, so they
        # should always agree, but if they diverge we re-align by name.
        idx_map = {n: i for i, n in enumerate(l2_names)}
        order   = [idx_map[n] for n in names if n in idx_map]
        l2_matrix = l2_matrix[order]

    q_l2 = query.astype(np.float64)
    qn   = np.linalg.norm(q_l2)
    if qn > 1e-12:
        q_l2 = q_l2 / qn
    q_d2 = _d2_pearson(query)

    sims_cos = l2_matrix @ q_l2.astype(np.float32)
    sims_d2  = d2_matrix @ q_d2
    sims     = _W_COSINE * sims_cos + _W_D2PEARSON * sims_d2

    # Dedupe by display name, keeping each mineral's best score.
    best_per_name: dict[str, float] = {}
    for i, n in enumerate(names):
        s = float(sims[i])
        if s > best_per_name.get(n, -np.inf):
            best_per_name[n] = s

    ranked = sorted(best_per_name.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for name, sim in ranked:
        meta = metadata[name]
        results.append({
            "mineral":        name,
            "formula":        meta.get("formula", ""),
            "similarity":     round(sim, 4),
            "confidence_pct": round(max(0.0, sim) * 100, 1),
            "source":         meta.get("source", "unknown"),
            "color":          meta.get("color", "#7c5c3e"),
            "description":    meta.get("description", ""),
            "peaks":          meta.get("peaks", []),
        })
    return results


def is_likely_mixture(results: list, primary_threshold: float = 0.78) -> bool:
    """
    Return True if the top match is too weak to be a pure-mineral
    identification. Empirically pure synthetic samples score above 0.95
    against the combined-metric matcher; pure RRUFF samples score 0.85+
    when their own RRUFF reference is in the DB. Anything below ~0.78 is
    likely either a mixture or an out-of-distribution sample.
    """
    if not results:
        return False
    return results[0]["similarity"] < primary_threshold

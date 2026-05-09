"""
Non-negative least squares mixture decomposition.

Decomposes a query spectrum as a non-negative linear combination of reference
spectra. Two design choices keep the result physically meaningful:

  1. Candidate selection — the full reference DB has ~2500 rows while spectra
     have 1951 points, so unconstrained NNLS over the whole basis is severely
     underdetermined and credit ends up sprayed across whichever obscure
     minerals happen to have peaks near the query's. We instead expose only:
       (a) the 25 curated synthetic refs (clean Lorentzians at canonical
           wavenumbers — a well-separated common-mineral basis), and
       (b) the top spectral-matching hits from RRUFF that aren't already
           covered by (a) — so rare-mineral queries can still be decomposed.

  2. Mixture detection — we only call something a mixture when the second
     component carries a non-trivial fraction (≥15% after re-normalisation),
     otherwise the decomposition is dominated by one phase and what looks
     like a 1–2 % "second mineral" is just NNLS expressing residual noise.
"""
import numpy as np
from scipy.optimize import nnls
from scipy.signal import savgol_filter
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database.reference_store import get_database, get_match_database
from database.synthetic_references import build_synthetic_db


# How many extra RRUFF candidates to add on top of the synthetic-25 basis.
_RRUFF_EXTRA_CANDIDATES = 8


def _d2_pearson_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    d2 = savgol_filter(v, window_length=15, polyorder=3, deriv=2)
    d2 -= d2.mean()
    n = np.linalg.norm(d2)
    return (d2 / n).astype(np.float32) if n > 1e-12 else d2.astype(np.float32)


def _build_candidate_set(query_l2: np.ndarray):
    """
    Returns (sub_matrix, sub_names, metadata) — the candidate basis for NNLS.

    The synthetic-25 are always present (canonical clean common-mineral basis).
    Additional RRUFF candidates are only included if their combined similarity
    to the query is *strictly higher* than the best synthetic candidate's,
    which signals "this query is a rare mineral the synthetic basis can't span"
    — otherwise adding RRUFF rows just gives NNLS spectral aliases of common
    minerals to split the credit across.
    """
    matrix,    names,    metadata = get_database()
    d2_matrix, d2_names, _        = get_match_database()
    if names != d2_names:
        idx_for = {n: i for i, n in enumerate(names)}
        order   = [idx_for[n] for n in d2_names if n in idx_for]
        matrix  = matrix[order]
        names   = d2_names

    synth_names = set(build_synthetic_db().keys())

    sims_cos = matrix @ query_l2.astype(np.float32)
    sims_d2  = d2_matrix @ _d2_pearson_vec(query_l2)
    sims     = 0.5 * sims_cos + 0.5 * sims_d2

    synth_idx = [i for i, n in enumerate(names) if n in synth_names]
    if not synth_idx:
        keep_idx = list(np.argsort(sims)[::-1][:_RRUFF_EXTRA_CANDIDATES])
    else:
        best_synth_sim = float(sims[synth_idx].max())
        keep_idx = list(synth_idx)

        rruff_order = [i for i in np.argsort(sims)[::-1] if names[i] not in synth_names]
        added = 0
        for i in rruff_order:
            if added >= _RRUFF_EXTRA_CANDIDATES:
                break
            if sims[i] <= best_synth_sim:
                break
            keep_idx.append(int(i))
            added += 1

    sub_names  = [names[i] for i in keep_idx]
    sub_matrix = matrix[keep_idx].astype(np.float64)
    return sub_matrix, sub_names, metadata


def decompose(query: np.ndarray, max_components: int = 5) -> dict:
    """
    Parameters
    ----------
    query : 1-D numpy array, L2-normalised, length GRID_POINTS.
    max_components : cap on how many minerals to report.

    Returns
    -------
    dict with keys:
        components: list of {mineral, formula, fraction, color}
        residual_norm: float  (lower = better fit)
        is_mixture: bool
    """
    q = query.astype(np.float64)
    qn = np.linalg.norm(q)
    if qn > 1e-12:
        q = q / qn

    sub_matrix, sub_names, metadata = _build_candidate_set(q)
    if sub_matrix.size == 0:
        return {"components": [], "residual_norm": 0.0, "is_mixture": False}
    if q.size != sub_matrix.shape[1]:
        return {"components": [], "residual_norm": 0.0, "is_mixture": False}

    coeffs, residual = nnls(sub_matrix.T, q)

    total = coeffs.sum()
    if total < 1e-12:
        return {"components": [], "residual_norm": float(residual), "is_mixture": False}

    fractions = coeffs / total

    # The reference DB may emit multiple rows per mineral (synthetic + RRUFF).
    # Aggregate fractions by display name so a single mineral never appears
    # twice in the output.
    aggregated: dict[str, float] = {}
    for i in range(len(sub_names)):
        aggregated[sub_names[i]] = aggregated.get(sub_names[i], 0.0) + float(fractions[i])

    above = [(n, f) for n, f in aggregated.items()
             if f >= config.MIXTURE_FRACTION_THRESHOLD]
    above.sort(key=lambda x: x[1], reverse=True)
    above = above[:max_components]

    if not above:
        return {"components": [], "residual_norm": float(residual), "is_mixture": False}

    kept_total = float(sum(f for _, f in above))
    components = []
    for name, frac in above:
        meta = metadata[name]
        components.append({
            "mineral": name,
            "formula": meta.get("formula", ""),
            "fraction": float(round(frac / kept_total, 4)),
            "fraction_pct": float(round(frac / kept_total * 100, 1)),
            "color": meta.get("color", "#ffffff"),
        })

    is_mixture = bool(len(components) > 1 and components[1]["fraction"] >= 0.15)

    return {
        "components": components,
        "residual_norm": round(float(residual), 6),
        "is_mixture": is_mixture,
    }

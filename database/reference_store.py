"""
Unified reference database.

For the 25 minerals where we have curated synthetic Lorentzian spectra, we
keep *both* the synthetic reference and the RRUFF average as separate matrix
rows. Synthetic refs win for clean queries (clean Lorentzian → high cosine
to clean Lorentzian); RRUFF refs win for real-world queries (noisy real
measurements → high cosine to noisy averaged real measurement). The matcher
deduplicates by mineral name afterwards, so the user only ever sees one row
per mineral but each query reaches whichever variant fits better.

Earlier versions either dropped one of the two when names collided (so
synthetic queries scored 79 % against RRUFF Quartz while the perfect
synthetic Quartz reference sat unused), or kept only synthetic (so RRUFF
queries had no ideal anchor). Keeping both fixes both regimes.

  MATCH_DB  — every reference in 2nd-derivative Pearson space, used by the
              spectral matcher.
  FULL_DB   — every reference in raw L2-normalised intensity space, used by
              NNLS mixture decomposition and reference-overlay rendering.
"""
import os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database.synthetic_references import load_or_build_synthetic_db

_match_cache = None   # (matrix, names, metadata, display_names)
_full_cache  = None   # (matrix, names, metadata, display_names)


def _d2_pearson_matrix(raw: np.ndarray) -> np.ndarray:
    """2nd-derivative → mean-centre → L2-normalise, row-wise."""
    from scipy.signal import savgol_filter
    d2 = savgol_filter(raw.astype(np.float64),
                       window_length=15, polyorder=3, deriv=2, axis=1)
    d2 -= d2.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(d2, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return (d2 / norms).astype(np.float32)


def _load_rruff_processed() -> dict:
    """Load + average RRUFF .npz files by mineral name."""
    if not os.path.isdir(config.PROCESSED_DIR):
        return {}
    groups:   dict[str, list]  = {}
    formulas: dict[str, str]   = {}
    for fname in os.listdir(config.PROCESSED_DIR):
        if not fname.endswith(".npz"):
            continue
        try:
            data = np.load(os.path.join(config.PROCESSED_DIR, fname), allow_pickle=True)
            name = str(data["mineral_name"]).strip()
            # RRUFF includes "unknown" / "unidentified" specimen entries —
            # they have no mineral identity, so letting them into the
            # reference matrix gives the matcher a wildcard that can outscore
            # the real top hit.
            if not name or name.lower() in {"unknown", "unidentified"}:
                continue
            spec = data["spectrum"].astype(np.float32)
            if len(spec) != config.GRID_POINTS or spec.max() < 0.01:
                continue
            groups.setdefault(name, []).append(spec)
            formulas.setdefault(name, str(data.get("formula", "")))
        except Exception:
            pass
    db = {}
    for name, spectra in groups.items():
        avg  = np.mean(spectra, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        if norm < 1e-10:
            continue
        db[name] = {
            "spectrum":    avg / norm,
            "formula":     formulas.get(name, ""),
            "n_samples":   len(spectra),
            "color":       "#58a6ff",
            "description": f"{len(spectra)} RRUFF sample{'s' if len(spectra)>1 else ''}",
            "peaks":       [],
            "source":      "rruff",
        }
    return db


def _build_rows():
    """
    Yield (row_key, display_name, entry_dict) for every reference row.

    Synthetic and RRUFF entries for the same mineral are emitted as TWO rows
    with distinct row_keys but the same display_name, so the matrix can carry
    both variants and the matcher dedupes at result time.
    """
    synth = load_or_build_synthetic_db()
    rruff = _load_rruff_processed()
    seen = set()
    # Synthetic first — guarantees stable ordering for the 25 well-known minerals.
    for name in sorted(synth):
        seen.add(name)
        yield (name, name, synth[name])
    for name in sorted(rruff):
        if name in seen:
            yield (f"{name}#rruff", name, rruff[name])
        else:
            yield (name, name, rruff[name])


def _build_db_arrays():
    rows = list(_build_rows())
    keys     = [r[0] for r in rows]
    display  = [r[1] for r in rows]
    spectra  = np.array([r[2]["spectrum"] for r in rows], dtype=np.float32)
    metadata = {r[1]: {k: v for k, v in r[2].items() if k != "spectrum"} for r in rows}
    # If both synth and RRUFF emit the same display name, prefer the synth
    # entry's metadata (which has color/description/peaks set explicitly).
    for r in rows:
        if r[2].get("source") == "synthetic":
            metadata[r[1]] = {k: v for k, v in r[2].items() if k != "spectrum"}
    return keys, display, spectra, metadata


def get_match_database() -> tuple:
    """
    Returns (matrix, display_names, metadata) — every reference in
    2nd-derivative Pearson space. Multiple rows can share a display_name
    (synthetic + RRUFF for the same mineral); the matcher dedupes results.
    """
    global _match_cache
    if _match_cache is not None:
        return _match_cache

    _, display, spectra, metadata = _build_db_arrays()
    matrix = _d2_pearson_matrix(spectra)
    _match_cache = (matrix, display, metadata)
    return _match_cache


def get_database() -> tuple:
    """
    Returns (matrix, display_names, metadata) — every reference, L2-normalised
    in intensity space. Multiple rows can share a display_name. NNLS callers
    that want a unique-per-mineral basis should aggregate by display_name.
    """
    global _full_cache
    if _full_cache is not None:
        return _full_cache

    _, display, spectra, metadata = _build_db_arrays()
    norms  = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-10, 1.0, norms)
    matrix = (spectra / norms).astype(np.float32)
    _full_cache = (matrix, display, metadata)
    return _full_cache


def invalidate_cache():
    global _match_cache, _full_cache
    _match_cache = _full_cache = None


def get_stats() -> dict:
    _, names, metadata = get_database()
    unique_names = list(dict.fromkeys(names))  # preserves order, dedupe
    sources = {}
    for n in unique_names:
        sources[metadata[n]["source"]] = sources.get(metadata[n]["source"], 0) + 1
    return {"total": len(unique_names), "sources": sources, "minerals": unique_names}

"""
Orchestrates: preprocess → identify → mixture.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessing import load_csv, preprocess
from core.spectral_matching import match, is_likely_mixture
from core.mixture import decompose
import core.cnn_model as cnn_model
import config


def run_full_analysis(file_bytes: bytes, mode: str = "auto") -> dict:
    """
    End-to-end analysis of an uploaded CSV spectrum.

    Parameters
    ----------
    file_bytes : raw bytes of uploaded file.
    mode : 'auto' | 'matching' | 'cnn'

    Returns
    -------
    Large result dict ready for JSON serialisation.
    """
    # 1. Load CSV
    wavenumber, intensity = load_csv(file_bytes)

    # 2. Preprocess
    grid, processed, raw_on_grid, steps = preprocess(wavenumber, intensity)
    processed_arr = np.array(processed, dtype=np.float32)

    # 3. Identify
    # Auto always uses spectral matching until CNN is trained on real (RRUFF) data.
    # CNN mode is only reliable after retraining on real spectra.
    use_cnn = (mode == "cnn") and cnn_model.CNN_AVAILABLE
    if use_cnn and not cnn_model.CNN_AVAILABLE:
        use_cnn = False

    if use_cnn:
        matches = cnn_model.predict(processed_arr)
        mode_used = "cnn"
    else:
        matches = match(processed_arr)
        mode_used = "matching"

    # 4. Mixture analysis
    mixture_result = decompose(processed_arr)

    # 5. Reference spectra for overlay (top 3 matches)
    reference_overlays = _get_reference_overlays(matches[:3], grid)

    return {
        "spectrum": {
            "wavenumber": grid,
            "raw": raw_on_grid,
            "processed": processed,
            "steps": steps,
        },
        "matches": matches,
        "mode_used": mode_used,
        "cnn_available": cnn_model.CNN_AVAILABLE,
        "is_likely_mixture": is_likely_mixture(matches),
        "mixture": mixture_result,
        "reference_overlays": reference_overlays,
    }


def _get_reference_overlays(matches: list, grid: list) -> list:
    from database.reference_store import get_database
    matrix, names, metadata = get_database()
    # Multiple rows can share a display name (synthetic + RRUFF). Pick the
    # synthetic one when both exist — its clean Lorentzian profile is what
    # users intuitively expect to see overlaid on their query.
    preferred_idx: dict[str, int] = {}
    for i, n in enumerate(names):
        if n not in preferred_idx:
            preferred_idx[n] = i
        elif metadata[n].get("source") == "synthetic":
            preferred_idx[n] = i

    overlays = []
    for match_entry in matches:
        name = match_entry["mineral"]
        if name in preferred_idx:
            ref_spec = matrix[preferred_idx[name]].tolist()
            overlays.append({
                "mineral": name,
                "spectrum": ref_spec,
                "color": match_entry.get("color", "#ffffff"),
            })
    return overlays

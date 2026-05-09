from flask import Blueprint, request, jsonify
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.store import get
from core.spectral_matching import match, is_likely_mixture
import core.cnn_model as cnn_model
import core.rf_model  as rf_model
from core.pipeline import _get_reference_overlays
import config

bp = Blueprint("identify", __name__)


# In auto mode the dispatcher consults every available identifier and returns
# the one with the highest top-1 confidence. The CNN can give very confident
# nonsense for spectra outside its training distribution (e.g. clean
# synthetic Lorentzians when trained on noisy RRUFF), so falling back to
# spectral matching whenever it scores higher prevents users from seeing
# obviously wrong predictions on demo samples.
_AUTO_CNN_MIN_CONFIDENCE = 0.55  # below this, prefer spectral matching


@bp.route("/identify", methods=["POST"])
def identify():
    data = request.get_json(force=True) or {}
    sid  = data.get("spectrum_id")
    mode = data.get("mode", "auto")

    if not sid:
        return jsonify({"error": "spectrum_id required."}), 400

    stored = get(sid)
    if stored is None:
        return jsonify({"error": "Spectrum not found. Upload first."}), 404

    query = np.array(stored["processed"], dtype=np.float32)
    grid  = stored["grid"]

    # Explicit CNN request but not available
    if mode == "cnn" and not cnn_model.CNN_AVAILABLE:
        return jsonify({"error": "CNN model not trained.",
                        "hint": "Run: python training/train_cnn.py",
                        "fallback": "matching"}), 503

    # Explicit RF request but not available
    if mode == "rf" and not rf_model.RF_AVAILABLE:
        return jsonify({"error": "RF model not trained.",
                        "hint": "Run: python training/train_rf.py",
                        "fallback": "matching"}), 503

    # Dispatch
    if mode == "cnn":
        matches   = cnn_model.predict(query)
        mode_used = "cnn"
    elif mode == "rf":
        matches   = rf_model.predict(query)
        mode_used = "rf"
    elif mode == "matching":
        matches   = match(query)
        mode_used = "matching"
    else:  # auto
        matches, mode_used = _auto_identify(query)

    reference_overlays = _get_reference_overlays(matches[:3], grid)

    return jsonify({
        "matches":            matches,
        "mode_used":          mode_used,
        "cnn_available":      cnn_model.CNN_AVAILABLE,
        "rf_available":       rf_model.RF_AVAILABLE,
        "is_likely_mixture":  is_likely_mixture(matches),
        "reference_overlays": reference_overlays,
    })


def _auto_identify(query: np.ndarray):
    """
    Auto-mode dispatch: try each available identifier and return the result
    with the highest top-1 confidence. Spectral matching is always run as a
    cheap safety net since the CNN/RF can be confidently wrong on
    out-of-distribution queries.
    """
    candidates = []  # list of (top1_score, mode_used, matches)

    spectral_matches = match(query)
    if spectral_matches:
        candidates.append((spectral_matches[0]["similarity"], "matching", spectral_matches))

    if cnn_model.CNN_AVAILABLE:
        try:
            cnn_matches = cnn_model.predict(query)
            if cnn_matches:
                # The CNN gives softmax probabilities; treat them as confidence.
                # We require a minimum threshold before letting the CNN win.
                top1 = cnn_matches[0]["similarity"]
                if top1 >= _AUTO_CNN_MIN_CONFIDENCE:
                    candidates.append((top1, "cnn", cnn_matches))
        except Exception:
            pass

    if rf_model.RF_AVAILABLE:
        try:
            rf_matches = rf_model.predict(query)
            if rf_matches:
                top1 = rf_matches[0]["similarity"]
                if top1 >= _AUTO_CNN_MIN_CONFIDENCE:
                    candidates.append((top1, "rf", rf_matches))
        except Exception:
            pass

    if not candidates:
        return spectral_matches, "matching"

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, mode_used, matches = candidates[0]
    return matches, mode_used

"""
Random Forest classifier for Raman spectra.
Pipeline: PCA (100 components) -> RandomForestClassifier.
RF_AVAILABLE is False until training/train_rf.py has been run.
"""
import os, json
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RF_AVAILABLE = False
_pipeline = None
_labels   = None

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


def _try_load():
    global RF_AVAILABLE, _pipeline, _labels
    if not _SKLEARN_OK:
        return
    if not (os.path.exists(config.RF_MODEL_PATH) and
            os.path.exists(config.LABEL_ENCODER_PATH)):
        return
    try:
        _pipeline = joblib.load(config.RF_MODEL_PATH)
        with open(config.LABEL_ENCODER_PATH) as f:
            _labels = json.load(f)["classes"]
        RF_AVAILABLE = True
        print(f"[RF] Loaded -- {len(_labels)} classes")
    except Exception as e:
        print(f"[RF] Load failed: {e}")

_try_load()


def predict(query: np.ndarray, top_k: int = None) -> list:
    if not RF_AVAILABLE:
        raise RuntimeError("RF model not available. Run: python training/train_rf.py")
    top_k = top_k or config.TOP_K
    x = query.reshape(1, -1)
    probs = _pipeline.predict_proba(x)[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    from database.reference_store import get_database
    _, _, metadata = get_database()
    results = []
    for idx in top_idx:
        mineral = _labels[idx]
        prob    = float(probs[idx])
        meta    = metadata.get(mineral, {})
        results.append({
            "mineral":        mineral,
            "formula":        meta.get("formula", ""),
            "similarity":     round(prob, 4),
            "confidence_pct": round(prob * 100, 1),
            "source":         "rf",
            "color":          meta.get("color", "#7c5c3e"),
            "description":    meta.get("description", ""),
            "peaks":          meta.get("peaks", []),
        })
    return results


def reload():
    global RF_AVAILABLE, _pipeline, _labels
    RF_AVAILABLE = False
    _pipeline = _labels = None
    _try_load()

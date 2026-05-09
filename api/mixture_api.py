from flask import Blueprint, request, jsonify
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.store import get
from core.mixture import decompose

bp = Blueprint("mixture", __name__)


@bp.route("/mixture", methods=["POST"])
def mixture():
    data = request.get_json(force=True) or {}
    sid = data.get("spectrum_id")
    if not sid:
        return jsonify({"error": "spectrum_id required."}), 400

    stored = get(sid)
    if stored is None:
        return jsonify({"error": "Spectrum not found. Upload first."}), 404

    query = np.array(stored["processed"], dtype=np.float32)
    result = decompose(query)
    return jsonify(result)

from flask import Blueprint, jsonify
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.reference_store import get_database
import config

bp = Blueprint("spectrum", __name__)


@bp.route("/spectrum/<mineral_name>", methods=["GET"])
def get_spectrum(mineral_name: str):
    matrix, names, metadata = get_database()
    if mineral_name not in names:
        return jsonify({"error": f"Mineral '{mineral_name}' not in database."}), 404

    # Synthetic refs are emitted first by reference_store, so the first
    # occurrence is the canonical (clean) reference when both variants exist.
    idx = names.index(mineral_name)
    meta = metadata[mineral_name]
    return jsonify({
        "mineral": mineral_name,
        "formula": meta.get("formula", ""),
        "source": meta.get("source", ""),
        "color": meta.get("color", "#ffffff"),
        "peaks": meta.get("peaks", []),
        "wavenumber": config.GRID.tolist(),
        "intensity": matrix[idx].tolist(),
    })

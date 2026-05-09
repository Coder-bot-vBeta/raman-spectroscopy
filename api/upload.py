from flask import Blueprint, request, jsonify
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.preprocessing import load_csv, preprocess
from api.store import save
import config
import numpy as np

bp = Blueprint("upload", __name__)


@bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    file_bytes = f.read()
    if len(file_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({"error": f"File too large (max {config.MAX_UPLOAD_MB} MB)."}), 413

    try:
        wavenumber, intensity = load_csv(file_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    try:
        grid, processed, raw_on_grid, steps = preprocess(wavenumber, intensity)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 500

    sid = save({
        "processed": processed,
        "raw": raw_on_grid,
        "grid": grid,
        "steps": steps,
        "filename": f.filename,
    })

    return jsonify({
        "spectrum_id": sid,
        "filename": f.filename,
        "n_points": len(grid),
        "wavenumber_range": [float(min(grid)), float(max(grid))],
        "spectrum": {
            "wavenumber": grid,
            "raw": raw_on_grid,
            "processed": processed,
            "steps": steps,
        },
    })

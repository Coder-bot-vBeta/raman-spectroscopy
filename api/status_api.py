from flask import Blueprint, jsonify
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.cnn_model as cnn_model
import core.rf_model  as rf_model
from database.reference_store import get_stats
import config

bp = Blueprint("status", __name__)


@bp.route("/status", methods=["GET"])
def status():
    stats       = get_stats()
    rruff_count = stats["sources"].get("rruff", 0)
    cnn_reliable = cnn_model.CNN_AVAILABLE and rruff_count > 0
    rf_reliable  = rf_model.RF_AVAILABLE   and rruff_count > 0

    return jsonify({
        "cnn_available":  cnn_model.CNN_AVAILABLE,
        "cnn_reliable":   cnn_reliable,
        "rf_available":   rf_model.RF_AVAILABLE,
        "rf_reliable":    rf_reliable,
        "n_references":   stats["total"],
        "rruff_count":    rruff_count,
        "sources":        stats["sources"],
        "grid": {
            "start":  config.GRID_START,
            "end":    config.GRID_END,
            "step":   config.GRID_STEP,
            "points": config.GRID_POINTS,
        },
    })

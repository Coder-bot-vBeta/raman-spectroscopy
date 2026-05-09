import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template
from flask_cors import CORS
import config

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

from api import upload_bp, identify_bp, mixture_bp, spectrum_bp, status_bp

app.register_blueprint(upload_bp,   url_prefix="/api")
app.register_blueprint(identify_bp, url_prefix="/api")
app.register_blueprint(mixture_bp,  url_prefix="/api")
app.register_blueprint(spectrum_bp, url_prefix="/api")
app.register_blueprint(status_bp,   url_prefix="/api")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    print("Starting Raman Mineral Analyzer...")
    print(f"  Reference DB: {config.SYNTHETIC_NPZ}")
    print(f"  CNN model:    {config.CNN_MODEL_PATH}")
    print("  Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=False, port=5000, host="0.0.0.0", use_reloader=False)

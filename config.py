import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Spectral grid: 100–4000 cm-1 at 2 cm-1 resolution → 1951 points
GRID_START = 100
GRID_END = 4000
GRID_STEP = 2
GRID = np.arange(GRID_START, GRID_END + GRID_STEP, GRID_STEP, dtype=np.float32)
GRID_POINTS = len(GRID)  # 1951

# Savitzky-Golay — window=5 on the 2 cm⁻¹ standard grid is 10 cm⁻¹ physical,
# matching the smoothing that was effectively applied to the stored RRUFF
# references (which were originally preprocessed at 0.5–1 cm⁻¹ native step).
# A wider window over-smooths sharp doublets like Gypsum's 1008/1017 cm⁻¹
# pair and merges them into one mis-positioned peak.
SAVGOL_WINDOW = 5
SAVGOL_POLYORDER = 3

# ALS baseline
ALS_LAMBDA = 1e5
ALS_P = 0.01
ALS_NITER = 10

# Paths
DATA_DIR = os.path.join(BASE_DIR, "database", "references")
RRUFF_RAW_DIR = os.path.join(DATA_DIR, "rruff")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_NPZ = os.path.join(DATA_DIR, "synthetic.npz")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_mineral.pt")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.json")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_mineral.joblib")

# Matching
TOP_K = 5
SIMILARITY_THRESHOLD = 0.60
MIXTURE_FRACTION_THRESHOLD = 0.05

# Upload
MAX_UPLOAD_MB = 16

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RRUFF_RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

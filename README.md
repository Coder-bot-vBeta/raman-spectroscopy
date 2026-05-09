# Raman Mineral Analyzer

A full-stack web application for automated mineral identification from Raman spectroscopy data. Upload a CSV spectrum and get back mineral identifications via three complementary methods: spectral matching, a 1D ResNet CNN, and a PCA+Random Forest classifier — with mixture decomposition support for multi-mineral samples.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Training Your Own Models](#training-your-own-models)
- [Adding New Reference Spectra](#adding-new-reference-spectra)
- [Potential Extensions](#potential-extensions)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Raman spectroscopy produces a unique spectral fingerprint for each mineral. This tool automates the identification step by:

1. **Preprocessing** the uploaded spectrum (Savitzky–Golay smoothing + ALS baseline removal, grid-normalized to 100–4000 cm⁻¹ at 2 cm⁻¹ resolution)
2. **Matching** against ~4,100 reference spectra from the [RRUFF database](https://rruff.info/) plus 25 built-in synthetic Lorentzian references
3. **Classifying** with a trained 1D ResNet CNN and/or a PCA+ExtraTrees Random Forest
4. **Decomposing mixtures** via Non-Negative Least Squares (NNLS) into constituent minerals with fractional abundances

The system was built as a mini-project for academic demonstration. The models are trained on RRUFF data with aggressive spectral augmentation, making them reasonably robust to real instrument noise.

---

## Features

- **Drag-and-drop CSV upload** with instant spectral preview (Plotly)
- **Three identification modes**: spectral matching, CNN, Random Forest — plus Auto mode (picks highest-confidence result)
- **Mixture decomposition**: decomposes a spectrum into 2+ minerals with percentage contributions
- **Reference overlay**: view the reference spectrum for any matched mineral alongside your upload
- **Dark-theme UI** with Bootstrap Icons and earth-tone palette
- **REST API** — all functionality is exposed via a Flask JSON API for programmatic access
- **25 built-in synthetic references** for the most common rock-forming minerals (no download required)
- **~4,100 RRUFF references** downloaded and processed at setup time

---

## Architecture

```
core/
  preprocessing.py     — CSV parsing, Savitzky–Golay + ALS baseline, grid normalization
  spectral_matching.py — Cosine similarity + 2nd-derivative Pearson dual-metric matcher
  cnn_model.py         — 1D ResNet inference wrapper (PyTorch)
  rf_model.py          — PCA + ExtraTrees inference wrapper (scikit-learn)
  mixture.py           — NNLS mixture decomposition
  pipeline.py          — End-to-end orchestration

api/
  upload.py            — POST /api/upload
  identify.py          — POST /api/identify
  mixture_api.py       — POST /api/mixture
  spectrum_api.py      — GET  /api/spectrum/<mineral>
  status_api.py        — GET  /api/status
  store.py             — In-memory spectrum store (UUID-keyed)

database/
  synthetic_references.py — 25 built-in Lorentzian peak tables
  rruff_parser.py         — Downloads + preprocesses RRUFF ZIP datasets
  reference_store.py      — Merged reference matrix (synthetic + RRUFF)

training/
  augmentation.py      — Noise, shift, baseline, broadening augmentations
  train_cnn.py         — 1D ResNet training script (PyTorch)
  train_rf.py          — PCA + ExtraTrees training script (scikit-learn)

models/
  cnn_mineral.pt       — Trained CNN checkpoint
  label_encoder.json   — Class label index

templates/index.html   — Single-page application
static/
  css/dark_theme.css
  js/main.js, plot.js, confidence.js
```

---

## Installation

### Prerequisites

- Python 3.10+
- ~2 GB disk space for RRUFF reference data download
- GPU optional (CPU inference is fast enough for a single spectrum)

### Steps

```bash
git clone https://github.com/<your-username>/raman-mineral-analyzer.git
cd raman-mineral-analyzer

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Download RRUFF Reference Spectra

The 4,069 preprocessed reference spectra are included in `database/references/processed/`. If you need to regenerate them from the raw RRUFF source (e.g., after cloning on a new machine):

```bash
python -c "from database.rruff_parser import download_and_process; download_and_process()"
```

This downloads ~300 MB of RRUFF ZIP archives, parses all spectra, and saves `.npz` files to `database/references/processed/`. It takes 5–15 minutes on a typical connection.

### Retrain the Random Forest (optional)

The RF model (`models/rf_mineral.joblib`) is excluded from this repository because it exceeds GitHub's 100 MB file limit. To regenerate it:

```bash
python training/train_rf.py
```

This takes ~10–20 minutes and produces `models/rf_mineral.joblib`. The system falls back to spectral matching and CNN if the RF model is absent.

---

## Usage

### Start the server

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Upload a spectrum

1. Drag a CSV file onto the upload area (or click to browse)
2. The CSV must have two columns: **wavenumber (cm⁻¹)** and **intensity** — any delimiter (comma, tab, semicolon, space) is accepted
3. Select an identification mode: **Auto**, **Spectral Matching**, **CNN**, or **Random Forest**
4. Click **Identify** to run analysis

### Try the demo samples

```bash
# Sample CSV files are in samples/
samples/quartz.csv
samples/calcite.csv
samples/pyrite.csv
samples/gypsum.csv
samples/quartz_calcite_mixture.csv   # 50:50 mixture for mixture decomposition
```

---

## API Reference

All endpoints return JSON. Errors return `{"error": "message"}` with an appropriate HTTP status code.

### `POST /api/upload`

Upload a spectrum CSV file.

**Request**: `multipart/form-data` with field `file`

**Response**:
```json
{
  "spectrum_id": "uuid-string",
  "wavenumbers": [100.0, 102.0, "..."],
  "intensities": [0.0, 0.02, "..."],
  "n_points": 1951
}
```

### `POST /api/identify`

Identify minerals in an uploaded spectrum.

**Request**:
```json
{
  "spectrum_id": "uuid-string",
  "mode": "auto",
  "top_k": 5
}
```

Modes: `auto` | `matching` | `cnn` | `rf`

**Response**:
```json
{
  "results": [
    {
      "mineral": "Quartz",
      "formula": "SiO₂",
      "confidence": 0.94,
      "source": "rruff",
      "color": "#d4a97a",
      "method": "matching"
    }
  ],
  "method_used": "auto"
}
```

### `POST /api/mixture`

Decompose a spectrum into a mineral mixture.

**Request**:
```json
{ "spectrum_id": "uuid-string" }
```

**Response**:
```json
{
  "components": [
    { "mineral": "Quartz", "fraction": 0.52, "confidence": 0.91 },
    { "mineral": "Calcite", "fraction": 0.48, "confidence": 0.87 }
  ]
}
```

### `GET /api/spectrum/<mineral_name>`

Retrieve the reference spectrum for a mineral.

**Response**:
```json
{
  "mineral": "Quartz",
  "formula": "SiO₂",
  "wavenumbers": [...],
  "intensities": [...],
  "peaks": [464, 206, 128],
  "color": "#d4a97a",
  "source": "rruff"
}
```

### `GET /api/status`

System health and model availability.

**Response**:
```json
{
  "cnn_available": true,
  "rf_available": false,
  "reference_count": 4069,
  "spectral_grid": { "start": 100, "end": 4000, "step": 2, "points": 1951 }
}
```

---

## Training Your Own Models

### CNN (1D ResNet)

```bash
python training/train_cnn.py
```

- Loads all `.npz` spectra from `database/references/processed/`
- Applies 80 augmentation variants per spectrum (noise, baseline, shift, broadening)
- 6-block 1D ResNet (32 channels, kernel=7) with global average pooling
- Saves checkpoint to `models/cnn_mineral.pt` and `models/label_encoder.json`
- Typical training time: 1–3 hours on CPU, 15–30 min on GPU

### Random Forest

```bash
python training/train_rf.py
```

- Same spectrum loading pipeline; 15 augmentation variants per spectrum
- PCA (120 components) → ExtraTreesClassifier (300 trees)
- Saves to `models/rf_mineral.joblib`
- Typical training time: 10–20 minutes on CPU

### Augmentation parameters

Edit `training/augmentation.py` to adjust:
- `noise_pct`: Gaussian noise level (default: 0.2–2.5%)
- `intensity_scale`: multiplicative jitter (default: 0.75–1.25×)
- `shift_cm`: wavenumber shift (default: ±10 cm⁻¹)

---

## Adding New Reference Spectra

### Synthetic references (no download)

Edit `database/synthetic_references.py`. Each mineral is defined as a list of Lorentzian peaks:

```python
"MyMineral": {
    "formula": "XY₂",
    "color": "#aabbcc",
    "description": "Brief description",
    "peaks": [
        {"center": 500, "amplitude": 1.0, "fwhm": 10},
        {"center": 850, "amplitude": 0.6, "fwhm": 8},
    ]
}
```

### Real RRUFF spectra

Additional minerals are automatically picked up if their `.npz` files are placed in `database/references/processed/`. Each `.npz` must contain:
- `wavenumbers` — 1D float array (1951 points, 100–4000 cm⁻¹)
- `intensities` — 1D float array (normalized)
- `metadata` — dict with keys `mineral`, `formula`, `source`

---

## Potential Extensions

This project has significant room for growth. Some high-value directions:

### Higher-accuracy models
- **Transformer encoder** on the 1951-point spectral sequence — the 1D ResNet is a strong baseline but attention over peak positions could improve peak-shape discrimination
- **Siamese networks** for few-shot mineral ID with very limited training data per class
- **Ensemble with uncertainty** — calibrated confidence from model disagreement as a quality flag

### Broader database coverage
- Integrate [WURM](https://wurm.info/) and [MRRUFF](https://mrruff.info/) databases alongside RRUFF for more diverse sample coverage
- Add **oriented crystal spectra** from RRUFF (currently only unoriented)
- Include **inclusion spectra** and **fluid inclusion** references for geological petrography

### Better preprocessing
- **Fluorescence rejection**: replace ALS with asymmetric Huber regression or iterative polynomial fitting
- **Cosmic ray removal**: automated spike detection before smoothing
- **Background subtraction** from multi-point baselines manually drawn in the UI

### Mixture decomposition
- Replace NNLS with **sparse Bayesian regression** (SBL) for automatic component count selection
- **Quantitative calibration** using synthetic mineral standards for weight-percent output
- **Mineral association rules** — co-occurrence priors from mineralogical literature (e.g., calcite rarely appears without dolomite in sedimentary rock)

### UI / UX
- **Batch processing** — upload a folder of CSVs, download an Excel report
- **Spectral editor** — interactive peak annotation before identification
- **Export to CIF** — output matched mineral crystal structures in CIF format for cross-reference with XRD
- **Mobile-responsive** redesign with touch-friendly spectrum zoom

### Deployment
- **Containerize** with Docker + Gunicorn for production deployment
- **GPU inference** via ONNX export of the CNN for faster throughput
- **Persistent spectrum store** — replace the in-memory UUID store with SQLite or Redis for multi-user sessions
- **Authentication** and per-user analysis history

### Instrument integration
- **Direct instrument import**: Renishaw `.wdf`, Horiba `.spe`, Ocean Optics `.txt` parsers
- **Real-time streaming** via WebSocket for instruments that can push spectra as they're acquired

---

## Project Structure

```
raman-mineral-analyzer/
├── app.py                    # Flask application entry point
├── config.py                 # Global constants (grid, thresholds, paths)
├── requirements.txt
├── README.md
│
├── core/                     # ML & spectral processing
│   ├── preprocessing.py
│   ├── spectral_matching.py
│   ├── cnn_model.py
│   ├── rf_model.py
│   ├── mixture.py
│   └── pipeline.py
│
├── api/                      # Flask REST endpoints
│   ├── upload.py
│   ├── identify.py
│   ├── mixture_api.py
│   ├── spectrum_api.py
│   ├── status_api.py
│   └── store.py
│
├── database/                 # Reference spectra & parsing
│   ├── synthetic_references.py
│   ├── rruff_parser.py
│   ├── reference_store.py
│   └── references/
│       ├── synthetic.npz             # 25 built-in Lorentzian references
│       └── processed/                # ~4,069 preprocessed RRUFF .npz files
│
├── training/                 # Model training scripts
│   ├── augmentation.py
│   ├── train_cnn.py
│   └── train_rf.py
│
├── models/                   # Trained model artifacts
│   ├── cnn_mineral.pt        # 1D ResNet checkpoint (~2.6 MB)
│   └── label_encoder.json    # Class labels
│
├── samples/                  # Demo CSV spectra
│   ├── quartz.csv
│   ├── calcite.csv
│   ├── pyrite.csv
│   ├── gypsum.csv
│   └── quartz_calcite_mixture.csv
│
├── static/
│   ├── css/dark_theme.css
│   └── js/
│       ├── main.js
│       ├── plot.js
│       └── confidence.js
│
└── templates/
    └── index.html
```

---

## Contributing

Pull requests are welcome. For large changes, open an issue first to discuss the direction.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request

---

## License

MIT License. See `LICENSE` for details.

---

*Built with Flask, PyTorch, scikit-learn, Plotly, and RRUFF database spectra.*

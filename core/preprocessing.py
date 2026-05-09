import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import io
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_csv(file_bytes: bytes):
    """Parse CSV bytes into (wavenumber, intensity) arrays."""
    for sep in [",", "\t", ";", " "]:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                sep=sep,
                comment="#",
                header=None,
                engine="python",
            )
            # Drop any non-numeric columns
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(axis=1, how="all").dropna(axis=0)
            if df.shape[1] >= 2 and len(df) > 10:
                wn = df.iloc[:, 0].values.astype(np.float32)
                inten = df.iloc[:, 1].values.astype(np.float32)
                # Sanity: wavenumbers should be roughly in [50, 5000]
                if 50 < wn.mean() < 5000:
                    return wn, inten
        except Exception:
            pass
    # Try with a header row
    for sep in [",", "\t", ";", " "]:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                sep=sep,
                comment="#",
                engine="python",
            )
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(axis=1, how="all").dropna(axis=0)
            if df.shape[1] >= 2 and len(df) > 10:
                wn = df.iloc[:, 0].values.astype(np.float32)
                inten = df.iloc[:, 1].values.astype(np.float32)
                if 50 < wn.mean() < 5000:
                    return wn, inten
        except Exception:
            pass
    raise ValueError(
        "Could not parse CSV. Expected two numeric columns: wavenumber, intensity."
    )


def remove_cosmic_rays(intensity: np.ndarray, spike_ratio: float = 5.0) -> np.ndarray:
    """
    Remove cosmic ray spikes.
    Cosmic rays are identified as single-point or narrow spikes significantly
    higher than their immediate neighbors. Real Raman peaks are broader, so
    their neighbors are also elevated. We never use a global threshold because
    dominant peaks (e.g., quartz 464 cm-1) would otherwise be falsely removed.
    """
    if len(intensity) < 5:
        return intensity.copy()
    result = intensity.copy()
    n = len(intensity)
    for i in range(2, n - 2):
        # Use a 5-point window; exclude the center point
        neighbor_vals = np.concatenate([intensity[i-2:i], intensity[i+1:i+3]])
        neighbor_max = neighbor_vals.max()
        if neighbor_max < 1e-10:
            continue
        # Only flag if center is much higher AND neighbors are all below it
        if (intensity[i] / neighbor_max > spike_ratio and
                intensity[i] > intensity[i-1] and
                intensity[i] > intensity[i+1]):
            result[i] = np.mean(neighbor_vals)
    return result


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline correction (Eilers & Boelens 2005)."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), format="csr")
    D = lam * D.T.dot(D)
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.diags(w, format="csr")
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return z


def savgol_smooth(intensity: np.ndarray, window: int = None, polyorder: int = None) -> np.ndarray:
    window = window or config.SAVGOL_WINDOW
    polyorder = polyorder or config.SAVGOL_POLYORDER
    # window must be odd and > polyorder
    if len(intensity) < window:
        window = max(polyorder + 2, 5)
        if window % 2 == 0:
            window += 1
    if window <= polyorder:
        return intensity.copy()
    return savgol_filter(intensity, window_length=window, polyorder=polyorder)


def normalize_l2(spectrum: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(spectrum)
    if norm < 1e-12:
        return spectrum.copy()
    return spectrum / norm


def preprocess(wavenumber: np.ndarray, intensity: np.ndarray):
    """
    Full preprocessing pipeline.

    The input is *first* interpolated onto the standard 2 cm⁻¹ grid, and only
    then smoothed / baseline-corrected. Doing it the other way round used to
    couple the S-G window length to the input grid spacing — a CSV sampled at
    4 cm⁻¹ saw a 44 cm⁻¹ smoothing window while RRUFF spectra (~0.5 cm⁻¹) saw
    ~5 cm⁻¹, and the resulting peak-broadening difference was enough to push
    the CNN from "Quartz 88 %" to "Iowaite 59 %" on the same underlying
    spectrum. Doing all the smoothing on the canonical grid means every
    spectrum — synthetic or RRUFF — sees the exact same S-G filter response.

    Returns:
        grid (list):       standard wavenumber grid
        processed (list):  preprocessed, L2-normalised intensity on grid
        raw_on_grid (list): raw intensity interpolated to grid (for display)
        steps (dict):      intermediate spectra for visualisation
    """
    # Sort (np.interp requires monotonically increasing x)
    idx = np.argsort(wavenumber)
    wn  = wavenumber[idx].astype(np.float64)
    raw_input = intensity[idx].astype(np.float64)

    # 1. Interpolate onto the standard grid first — this is the key fix.
    grid = config.GRID.astype(np.float64)
    raw_on_grid = np.interp(grid, wn, raw_input, left=0.0, right=0.0)

    inten = raw_on_grid.copy()

    # 2. Cosmic ray removal (now on uniform grid)
    inten = remove_cosmic_rays(inten)

    # 3. Savitzky-Golay smoothing (window length is now in standard grid steps)
    smoothed = savgol_smooth(inten)
    inten = smoothed

    # 4. ALS baseline correction (uniform grid → λ has consistent meaning)
    baseline = als_baseline(inten, lam=config.ALS_LAMBDA, p=config.ALS_P, niter=config.ALS_NITER)
    corrected = inten - baseline
    inten = corrected

    # 5. Clamp negatives
    inten = np.clip(inten, 0, None)

    # 6. L2 normalise
    processed_norm = normalize_l2(inten)

    steps = {
        "raw":       raw_on_grid.tolist(),
        "smoothed":  smoothed.tolist(),
        "baseline":  baseline.tolist(),
        "corrected": inten.tolist(),
    }

    return grid.tolist(), processed_norm.tolist(), raw_on_grid.tolist(), steps

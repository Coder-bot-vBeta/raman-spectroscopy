"""
Spectral augmentation for CNN training.
Simulates real instrument variation: noise, baseline drift, wavenumber shift,
intensity jitter, and peak broadening.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def augment(spectrum: np.ndarray, n_variants: int = 80,
            rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = len(spectrum)
    variants = np.empty((n_variants, n), dtype=np.float32)

    for i in range(n_variants):
        s = spectrum.astype(np.float64)

        # 1. Gaussian noise (instrument detector noise)
        sigma = rng.uniform(0.002, 0.025)
        s += rng.normal(0, sigma, n)

        # 2. Intensity scale jitter (laser power fluctuation)
        s *= rng.uniform(0.75, 1.25)

        # 3. Wavenumber shift (calibration drift ±10 cm-1)
        shift = int(rng.integers(-5, 6))   # steps of 2 cm-1
        if shift:
            s = np.roll(s, shift)
            if shift > 0: s[:shift] = 0
            else:         s[shift:] = 0

        # 4. Polynomial baseline (residual fluorescence after ALS)
        # Small amplitude, applied less often — the inference query has had
        # its baseline removed by ALS, so over-baselining training samples
        # widens the train/inference distribution gap.
        if rng.random() > 0.7:
            degree = rng.integers(1, 4)
            coeffs = rng.normal(0, 0.01, degree + 1)
            x = np.linspace(-1, 1, n)
            baseline = np.polyval(coeffs, x)
            s += baseline * s.max()

        # 5. Peak broadening via simple moving average (fast, no scipy)
        if rng.random() > 0.6:
            w = int(rng.integers(2, 5))
            kernel = np.ones(w) / w
            s = np.convolve(s, kernel, mode='same')

        # 6. Random spectral dropout (detector artefacts)
        if rng.random() > 0.8:
            n_drop = rng.integers(1, 5)
            drop_pos = rng.integers(0, n, n_drop)
            s[drop_pos] = 0

        # 7. Clamp and L2 normalise
        s = np.clip(s, 0, None)
        norm = np.linalg.norm(s)
        if norm > 1e-12:
            s /= norm

        variants[i] = s.astype(np.float32)

    return variants

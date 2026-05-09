"""
Generate sample CSV spectrum files for testing.

Noise level kept low (0.3 % of peak height) so the L2-normalised demo
spectra don't acquire a large spurious "noise floor" outside the peaks —
that floor was previously enough to push the CNN off the correct mineral.
Peak lists mirror synthetic_references.py so OH/water bands above 3000 cm⁻¹
are present where the reference DB expects them.
"""
import numpy as np
import os

GRID = np.arange(100, 4001, 4, dtype=np.float32)

def lorentzian(x, center, amplitude, fwhm):
    gamma = fwhm / 2.0
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

SAMPLES = {
    "quartz": {
        "peaks": [(128, 1.0, 20), (206, 0.35, 18), (265, 0.15, 15), (394, 0.10, 22),
                  (464, 10.0, 12), (696, 0.05, 20), (808, 0.05, 28), (1082, 0.07, 28)],
    },
    "calcite": {
        "peaks": [(155, 0.07, 18), (282, 0.10, 18), (712, 0.04, 22), (1086, 1.0, 16)],
    },
    "pyrite": {
        "peaks": [(343, 0.50, 10), (380, 1.0, 9), (430, 0.30, 14)],
    },
    "gypsum": {
        # Includes the OH stretches at 3406 / 3490 cm⁻¹ — without them the
        # spectrum doesn't match the synthetic Gypsum reference and the CNN
        # falls back to whatever single-peak class it thinks fits best.
        "peaks": [(415, 0.06, 18), (493, 0.08, 16), (618, 0.03, 18), (671, 0.04, 18),
                  (1008, 1.0, 12), (1017, 0.80, 12), (3406, 0.20, 28), (3490, 0.20, 28)],
    },
}

NOISE_PCT = 0.003  # 0.3 % of peak height — realistic, but doesn't dominate L2.

rng = np.random.default_rng(0)

out_dir = os.path.dirname(os.path.abspath(__file__))
for name, data in SAMPLES.items():
    spectrum = np.zeros(len(GRID), dtype=np.float32)
    for c, a, w in data["peaks"]:
        spectrum += lorentzian(GRID, c, a, w)
    spectrum += rng.normal(0, NOISE_PCT * spectrum.max(), len(GRID)).astype(np.float32)
    spectrum = np.clip(spectrum, 0, None)

    path = os.path.join(out_dir, f"{name}.csv")
    rows = "\n".join(f"{wn:.2f},{inten:.4f}" for wn, inten in zip(GRID, spectrum))
    with open(path, "w") as f:
        f.write("wavenumber,intensity\n" + rows)
    print(f"Saved {path}")

# Mixture: quartz + 0.4·calcite
mix = np.zeros(len(GRID), dtype=np.float32)
for c, a, w in SAMPLES["quartz"]["peaks"]:
    mix += lorentzian(GRID, c, a, w)
for c, a, w in SAMPLES["calcite"]["peaks"]:
    mix += 0.4 * lorentzian(GRID, c, a, w)
mix += rng.normal(0, NOISE_PCT * mix.max(), len(GRID)).astype(np.float32)
mix = np.clip(mix, 0, None)
rows = "\n".join(f"{wn:.2f},{inten:.4f}" for wn, inten in zip(GRID, mix))
path = os.path.join(out_dir, "quartz_calcite_mixture.csv")
with open(path, "w") as f:
    f.write("wavenumber,intensity\n" + rows)
print(f"Saved {path}")

# Equal-weight Quartz + Calcite (the harder case for identification)
mix2 = np.zeros(len(GRID), dtype=np.float32)
for c, a, w in SAMPLES["quartz"]["peaks"]:
    mix2 += lorentzian(GRID, c, a, w)
for c, a, w in SAMPLES["calcite"]["peaks"]:
    mix2 += lorentzian(GRID, c, a, w)
mix2 += rng.normal(0, NOISE_PCT * mix2.max(), len(GRID)).astype(np.float32)
mix2 = np.clip(mix2, 0, None)
rows = "\n".join(f"{wn:.2f},{inten:.4f}" for wn, inten in zip(GRID, mix2))
path = os.path.join(out_dir, "quartz_calcite_equal.csv")
with open(path, "w") as f:
    f.write("wavenumber,intensity\n" + rows)
print(f"Saved {path}")

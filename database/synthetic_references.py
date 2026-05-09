"""
Built-in synthetic reference spectra for 25 common minerals.
Generated from published Raman peak tables using Lorentzian profiles.
These are available immediately without any data download.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Each entry: mineral_name -> {formula, peaks: [(center_cm1, rel_amplitude, fwhm_cm1), ...], color, description}
MINERAL_PEAKS = {
    "Quartz": {
        "formula": "SiO2",
        "color": "#74b9ff",
        "description": "Most abundant mineral in Earth's crust",
        "peaks": [(128, 1.0, 20), (206, 0.35, 18), (265, 0.15, 15), (394, 0.10, 22),
                  (464, 10.0, 12), (696, 0.05, 20), (808, 0.05, 28), (1082, 0.07, 28)],
    },
    "Calcite": {
        "formula": "CaCO3",
        "color": "#a29bfe",
        "description": "Common carbonate mineral in limestone and marble",
        "peaks": [(155, 0.07, 18), (282, 0.10, 18), (712, 0.04, 22), (1086, 1.0, 16)],
    },
    "Dolomite": {
        "formula": "CaMg(CO3)2",
        "color": "#fd79a8",
        "description": "Carbonate sedimentary mineral",
        "peaks": [(176, 0.06, 18), (300, 0.10, 18), (724, 0.04, 22), (1097, 1.0, 16)],
    },
    "Gypsum": {
        "formula": "CaSO4·2H2O",
        "color": "#ffeaa7",
        "description": "Soft sulfate mineral used in plaster",
        "peaks": [(415, 0.06, 18), (493, 0.08, 16), (618, 0.03, 18), (671, 0.04, 18),
                  (1008, 1.0, 12), (1017, 0.80, 12), (3406, 0.20, 28), (3490, 0.20, 28)],
    },
    "Pyrite": {
        "formula": "FeS2",
        "color": "#fdcb6e",
        "description": "Iron sulfide, fools gold",
        "peaks": [(343, 0.50, 10), (380, 1.0, 9), (430, 0.30, 14)],
    },
    "Hematite": {
        "formula": "α-Fe2O3",
        "color": "#e17055",
        "description": "Primary iron ore, gives red pigment",
        "peaks": [(225, 0.40, 18), (245, 0.10, 18), (292, 1.0, 22), (299, 0.90, 18),
                  (412, 0.20, 22), (498, 0.10, 22), (613, 0.20, 28), (1320, 0.10, 45)],
    },
    "Magnetite": {
        "formula": "Fe3O4",
        "color": "#636e72",
        "description": "Magnetic iron oxide mineral",
        "peaks": [(193, 0.20, 28), (305, 0.30, 28), (540, 0.15, 38), (670, 1.0, 28)],
    },
    "Goethite": {
        "formula": "α-FeOOH",
        "color": "#b8860b",
        "description": "Iron oxyhydroxide, common in soils",
        "peaks": [(246, 1.0, 38), (300, 0.50, 38), (385, 0.30, 35), (480, 0.20, 38),
                  (546, 0.15, 38), (686, 0.10, 38)],
    },
    "Malachite": {
        "formula": "Cu2(CO3)(OH)2",
        "color": "#00b894",
        "description": "Green copper carbonate mineral",
        "peaks": [(154, 0.20, 18), (178, 0.15, 16), (268, 0.10, 18), (355, 1.0, 16),
                  (433, 0.50, 18), (539, 0.10, 18), (1099, 0.80, 22), (1370, 0.10, 28),
                  (1494, 0.30, 22)],
    },
    "Azurite": {
        "formula": "Cu3(CO3)2(OH)2",
        "color": "#0984e3",
        "description": "Blue copper carbonate mineral",
        "peaks": [(181, 0.20, 18), (249, 0.15, 18), (399, 1.0, 16), (404, 0.90, 16),
                  (770, 0.10, 28), (838, 0.10, 28), (1098, 0.50, 22), (1457, 0.10, 28)],
    },
    "Fluorite": {
        "formula": "CaF2",
        "color": "#6c5ce7",
        "description": "Halide mineral used as flux",
        "peaks": [(322, 1.0, 18), (420, 0.10, 38)],
    },
    "Diamond": {
        "formula": "C",
        "color": "#dfe6e9",
        "description": "Hardest natural mineral",
        "peaks": [(1332, 1.0, 5)],
    },
    "Graphite": {
        "formula": "C",
        "color": "#2d3436",
        "description": "Layered carbon polymorph",
        "peaks": [(1350, 0.50, 28), (1582, 1.0, 22), (2700, 0.80, 55)],
    },
    "Rutile": {
        "formula": "TiO2",
        "color": "#e84393",
        "description": "High-refractive index titanium oxide",
        "peaks": [(143, 1.0, 14), (235, 0.10, 28), (447, 0.30, 28), (612, 0.20, 22)],
    },
    "Anatase": {
        "formula": "TiO2",
        "color": "#fd79a8",
        "description": "Metastable TiO2 polymorph",
        "peaks": [(144, 1.0, 14), (197, 0.10, 18), (399, 0.30, 22), (515, 0.20, 22),
                  (639, 0.40, 22)],
    },
    "Zircon": {
        "formula": "ZrSiO4",
        "color": "#55efc4",
        "description": "Common accessory mineral in igneous rocks",
        "peaks": [(214, 0.10, 18), (357, 0.20, 18), (439, 0.10, 18), (975, 1.0, 14),
                  (1008, 0.30, 18)],
    },
    "Apatite": {
        "formula": "Ca5(PO4)3(F,Cl,OH)",
        "color": "#00cec9",
        "description": "Phosphate mineral, component of bones and teeth",
        "peaks": [(432, 0.20, 22), (580, 0.10, 22), (607, 0.20, 22), (965, 1.0, 18),
                  (1040, 0.40, 28)],
    },
    "Barite": {
        "formula": "BaSO4",
        "color": "#b2bec3",
        "description": "Heavy barium sulfate mineral",
        "peaks": [(139, 0.40, 14), (453, 0.50, 18), (617, 0.20, 18), (984, 1.0, 14),
                  (1136, 0.20, 22)],
    },
    "Orthoclase": {
        "formula": "KAlSi3O8",
        "color": "#e17055",
        "description": "Potassium feldspar, common in granites",
        "peaks": [(285, 0.30, 28), (476, 0.50, 28), (508, 1.0, 28), (761, 0.20, 38)],
    },
    "Forsterite": {
        "formula": "Mg2SiO4",
        "color": "#6ab04c",
        "description": "Magnesium-rich olivine end-member",
        "peaks": [(224, 0.10, 18), (304, 0.10, 18), (585, 0.20, 22), (824, 1.0, 18),
                  (857, 0.70, 18), (964, 0.40, 22)],
    },
    "Sulfur": {
        "formula": "S8",
        "color": "#f9ca24",
        "description": "Native element found near volcanic vents",
        "peaks": [(153, 0.50, 14), (219, 1.0, 14), (473, 0.30, 18)],
    },
    "Galena": {
        "formula": "PbS",
        "color": "#747d8c",
        "description": "Primary lead ore mineral",
        "peaks": [(145, 1.0, 14), (432, 0.20, 28)],
    },
    "Sphalerite": {
        "formula": "ZnS",
        "color": "#eccc68",
        "description": "Primary zinc ore mineral",
        "peaks": [(271, 0.50, 14), (352, 1.0, 14)],
    },
    "Talc": {
        "formula": "Mg3Si4O10(OH)2",
        "color": "#a4b0be",
        "description": "Softest mineral, used in cosmetics",
        "peaks": [(186, 0.10, 22), (290, 0.20, 28), (364, 0.40, 22), (674, 1.0, 22),
                  (1048, 0.30, 22), (3677, 0.80, 18)],
    },
    "Kaolinite": {
        "formula": "Al2Si2O5(OH)4",
        "color": "#ffe0ac",
        "description": "Clay mineral from weathered feldspar",
        "peaks": [(143, 0.10, 28), (247, 0.10, 28), (432, 0.40, 28), (463, 0.30, 28),
                  (701, 0.10, 28), (754, 0.10, 28), (913, 0.20, 22), (937, 0.20, 22),
                  (3620, 0.60, 18), (3695, 1.0, 18)],
    },
}


def _lorentzian(x: np.ndarray, center: float, amplitude: float, fwhm: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return amplitude * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def generate_spectrum(peaks: list, grid: np.ndarray) -> np.ndarray:
    spectrum = np.zeros(len(grid), dtype=np.float32)
    for center, amplitude, fwhm in peaks:
        if config.GRID_START <= center <= config.GRID_END:
            spectrum += _lorentzian(grid, center, amplitude, fwhm).astype(np.float32)
    max_val = spectrum.max()
    if max_val > 0:
        spectrum /= max_val
    return spectrum


def build_synthetic_db() -> dict:
    """Return dict: mineral_name -> {'spectrum', 'formula', 'color', 'description', 'peaks', 'source'}"""
    grid = config.GRID
    db = {}
    for mineral, data in MINERAL_PEAKS.items():
        spec = generate_spectrum(data["peaks"], grid)
        norm = np.linalg.norm(spec)
        db[mineral] = {
            "spectrum": spec / norm if norm > 0 else spec,
            "formula": data["formula"],
            "color": data["color"],
            "description": data["description"],
            "peaks": [p[0] for p in data["peaks"]],
            "source": "synthetic",
        }
    return db


def save_synthetic_db(path: str = None):
    path = path or config.SYNTHETIC_NPZ
    db = build_synthetic_db()
    names = list(db.keys())
    spectra = np.array([db[n]["spectrum"] for n in names], dtype=np.float32)
    formulas = [db[n]["formula"] for n in names]
    np.savez(path, names=names, spectra=spectra, formulas=formulas)
    return db


def load_or_build_synthetic_db(path: str = None) -> dict:
    path = path or config.SYNTHETIC_NPZ
    if not os.path.exists(path):
        return save_synthetic_db(path)
    data = np.load(path, allow_pickle=True)
    db = {}
    for name, spec, formula in zip(data["names"], data["spectra"], data["formulas"]):
        db[str(name)] = {
            "spectrum": spec,
            "formula": str(formula),
            "color": MINERAL_PEAKS.get(str(name), {}).get("color", "#ffffff"),
            "description": MINERAL_PEAKS.get(str(name), {}).get("description", ""),
            "peaks": [p[0] for p in MINERAL_PEAKS.get(str(name), {}).get("peaks", [])],
            "source": "synthetic",
        }
    return db

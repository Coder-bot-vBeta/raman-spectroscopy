"""
Download and parse RRUFF Raman spectra using RamanSPy's parser.

Usage:
    python -m database.rruff_parser [--datasets excellent_unoriented fair_unoriented]

Downloads ZIP files from rruff.net, parses each .txt spectrum,
preprocesses and saves as .npz files to PROCESSED_DIR.
"""
import os, sys, io, zipfile, argparse
import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.preprocessing import preprocess

# Correct domain is rruff.net, not rruff.info
BASE_URL = "https://rruff.net/zipped_data_files/raman/{dataset}.zip"

AVAILABLE_DATASETS = {
    "fair_unoriented":      59,    # MB
    "excellent_unoriented": 241,
    "excellent_oriented":   77,
    "LR-Raman":             227,
    "fair_oriented":        1,
}


def download_zip(dataset: str) -> bytes:
    url = BASE_URL.format(dataset=dataset)
    print(f"Downloading {url}  ({AVAILABLE_DATASETS.get(dataset, '?')} MB) ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pb:
        for chunk in r.iter_content(chunk_size=131072):
            buf.write(chunk)
            pb.update(len(chunk))
    raw = buf.getvalue()
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise ValueError(f"Downloaded content is not a ZIP (got {len(raw)} bytes). URL may have changed.")
    return raw


def _parse_one(file_obj) -> dict | None:
    """Parse a single RRUFF .txt file; returns dict or None."""
    metadata = {}
    wavenumbers, intensities = [], []
    try:
        for raw_line in file_obj.readlines():
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line or line.startswith("##END"):
                if line.startswith("##END"):
                    break
                continue
            if line.startswith("##"):
                idx = line.find("=")
                if idx > 0:
                    metadata[line[2:idx].strip().upper()] = line[idx+1:].strip()
            else:
                parts = line.replace(";", ",").split(",")
                if len(parts) >= 2:
                    try:
                        wavenumbers.append(float(parts[0]))
                        intensities.append(float(parts[1]))
                    except ValueError:
                        pass
    except Exception:
        return None

    if len(wavenumbers) < 20 or "NAMES" not in metadata:
        return None

    return {
        "mineral_name": metadata["NAMES"].split(",")[0].strip(),
        "rruff_id":     metadata.get("RRUFFID", ""),
        "formula":      metadata.get("IDEAL CHEMISTRY", ""),
        "wavenumber":   np.array(wavenumbers, dtype=np.float32),
        "intensity":    np.array(intensities,  dtype=np.float32),
    }


def process_zip(raw_zip: bytes) -> int:
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    saved = 0
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        entries = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        print(f"Parsing {len(entries)} spectrum files ...")
        for fname in tqdm(entries, unit="spec"):
            try:
                parsed = _parse_one(zf.open(fname))
                if parsed is None:
                    continue
                _, processed, _, _ = preprocess(parsed["wavenumber"], parsed["intensity"])
                out = os.path.join(
                    config.PROCESSED_DIR,
                    f"{parsed['mineral_name'].replace(' ', '_')}_{parsed['rruff_id']}.npz",
                )
                np.savez(out,
                    mineral_name=parsed["mineral_name"],
                    rruff_id=parsed["rruff_id"],
                    formula=parsed["formula"],
                    spectrum=np.array(processed, dtype=np.float32),
                )
                saved += 1
            except Exception:
                pass
    return saved


def run(datasets: list):
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    total_saved = 0
    for ds in datasets:
        try:
            raw = download_zip(ds)
            n = process_zip(raw)
            total_saved += n
            print(f"  {ds}: saved {n} spectra")
        except Exception as e:
            print(f"  {ds}: FAILED — {e}")

    from database.reference_store import invalidate_cache
    invalidate_cache()
    print(f"\nTotal saved: {total_saved} spectra -> {config.PROCESSED_DIR}")
    if total_saved > 0:
        print("Now retrain: python training/train_cnn.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["fair_unoriented", "excellent_unoriented"],
                    choices=list(AVAILABLE_DATASETS.keys()))
    args = ap.parse_args()
    run(args.datasets)

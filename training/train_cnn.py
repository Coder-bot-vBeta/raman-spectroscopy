"""
Train the 1D ResNet on GPU with PyTorch.

Usage:
    python training/train_cnn.py [--epochs 100] [--variants 80] [--batch 512] [--min_spectra 2]

--min_spectra : only train on minerals that have >= N real spectra in the DB.
                Minerals with a single spectrum are excluded to reduce noise.
"""
import os, sys, json, argparse, time, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch not found.  pip install torch"); sys.exit(1)

from training.augmentation import augment
from core.cnn_model         import build_model


def load_all_spectra(min_spectra: int = 2):
    """
    Load every individual .npz file from PROCESSED_DIR.
    Returns (groups, class_names) where groups maps mineral name -> list of spectra.
    Filters out minerals with fewer than min_spectra samples.

    The 25 curated synthetic spectra are added to *every* matching class
    (not just classes RRUFF doesn't cover) so the CNN sees clean Lorentzian
    profiles alongside real noisy measurements — without this the model
    never observes the synthetic distribution and fails on any clean query
    (the demo "Quartz" button used to predict Iowaite).
    """
    # Collect all files grouped by mineral name
    groups: dict[str, list[np.ndarray]] = {}
    for fpath in glob.glob(os.path.join(config.PROCESSED_DIR, "*.npz")):
        try:
            d = np.load(fpath, allow_pickle=True)
            name = str(d["mineral_name"])
            spec = d["spectrum"].astype(np.float32)
            if len(spec) != config.GRID_POINTS:
                continue
            groups.setdefault(name, []).append(spec)
        except Exception:
            pass

    # Add synthetic spectra to ALL classes — both RRUFF-overlapping (extra
    # clean sample) and synthetic-only (their single training sample).
    from database.synthetic_references import build_synthetic_db
    synth = build_synthetic_db()
    n_synth_added = 0
    for name, entry in synth.items():
        groups.setdefault(name, []).append(entry["spectrum"].astype(np.float32))
        n_synth_added += 1
    print(f"  Added {n_synth_added} synthetic spectra into the training pool")

    # Filter by minimum sample count. Synthetic-only classes have 1 sample;
    # default min_spectra=2 still drops them, so the user must pass
    # --min_spectra 1 to include them. Synthetic-augmented RRUFF classes have
    # ≥3 samples and remain.
    if min_spectra > 1:
        before = len(groups)
        groups = {k: v for k, v in groups.items() if len(v) >= min_spectra}
        print(f"  Filtered {before} -> {len(groups)} minerals (min_spectra={min_spectra})")

    class_names = sorted(groups.keys())
    print(f"  Classes: {len(class_names)}   Total spectra: {sum(len(v) for v in groups.values())}")
    return groups, class_names


def train(epochs=100, variants_per_class=25, batch_size=256, lr=1e-3, min_spectra=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("\nLoading spectra ...")
    groups, class_names = load_all_spectra(min_spectra)
    n_classes = len(class_names)

    # Pre-compute augmented dataset — fast numpy, fits in RAM at variants<=25
    print(f"Augmenting ({variants_per_class} variants per real spectrum) ...")
    rng = np.random.default_rng(42)
    X_list, y_list = [], []
    for i, name in enumerate(class_names):
        for spec in groups[name]:
            variants = augment(spec, n_variants=variants_per_class, rng=rng)
            X_list.append(variants)
            y_list.extend([i] * variants_per_class)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_classes}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"Dataset: {X.shape}  RAM: {X.nbytes/1e9:.2f} GB")

    perm  = rng.permutation(len(X))
    X, y  = X[perm], y[perm]
    split = int(0.9 * len(X))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    print(f"Train: {len(X_tr):,}   Val: {len(X_val):,}")

    ds_tr  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dl_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,
                        pin_memory=(device.type=="cuda"), num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size*2, shuffle=False,
                        pin_memory=(device.type=="cuda"), num_workers=0)

    model = build_model(config.GRID_POINTS, n_classes).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(dl_tr),
        pct_start=0.1, anneal_strategy="cos",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience_ctr = 0
    patience     = 12
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Acc':>9}  {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = correct = seen = 0

        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)

            # Mixup augmentation
            lam = float(np.random.beta(0.4, 0.4)) if np.random.rand() > 0.5 else 1.0
            if lam < 1.0:
                idx  = torch.randperm(len(xb), device=device)
                xb   = lam * xb + (1 - lam) * xb[idx]
                yb_m = yb[idx]
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb_m)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss   = criterion(logits, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()
            seen       += len(xb)

        train_loss = total_loss / seen
        train_acc  = correct / seen

        model.eval()
        val_correct = val_seen = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                val_correct += (model(xb).argmax(1) == yb).sum().item()
                val_seen    += len(xb)
        val_acc = val_correct / val_seen
        elapsed = time.time() - t0

        marker = " << best" if val_acc > best_val_acc else ""
        print(f"{epoch:>6}  {train_loss:>11.4f}  {train_acc:>9.2%}  {val_acc:>8.2%}  {elapsed:>4.1f}s{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({
                "model_state": model.state_dict(),
                "arch":        "RamanResNet",
                "n_classes":   n_classes,
                "grid_points": config.GRID_POINTS,
                "val_acc":     best_val_acc,
            }, config.CNN_MODEL_PATH)
            # Save label encoder alongside every best model so a crash mid-training
            # still leaves a consistent checkpoint + labels pair.
            with open(config.LABEL_ENCODER_PATH, "w") as f:
                json.dump({"classes": class_names}, f, indent=2)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stop (patience={patience})")
                break

    print(f"\nBest val accuracy : {best_val_acc:.2%}")
    print(f"Classes           : {n_classes}")
    print(f"Model saved       : {config.CNN_MODEL_PATH}")

    import core.cnn_model as cnn_mod
    cnn_mod.reload()
    print("CNN reloaded -- restart Flask to use the new model.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",       type=int,   default=100)
    ap.add_argument("--variants",     type=int,   default=80)
    ap.add_argument("--batch",        type=int,   default=512)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--min_spectra",  type=int,   default=2,
                    help="Only train on minerals with >= N real spectra")
    args = ap.parse_args()
    train(epochs=args.epochs, variants_per_class=args.variants,
          batch_size=args.batch, lr=args.lr, min_spectra=args.min_spectra)

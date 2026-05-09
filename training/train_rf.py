"""
Train a PCA + Random Forest classifier on Raman spectra.

Usage:
    python training/train_rf.py [--trees 300] [--components 120] [--min_spectra 2]

Shares the same load_all_spectra() + augment() pipeline as train_cnn.py.
Much faster to train than the ResNet; good accuracy on real spectra.
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
except ImportError:
    print("scikit-learn / joblib not found.  pip install scikit-learn joblib"); sys.exit(1)

from training.train_cnn  import load_all_spectra
from training.augmentation import augment


def train(n_trees: int = 300, n_components: int = 120,
          variants_per_class: int = 15, min_spectra: int = 2):

    print("\nLoading spectra ...")
    groups, class_names = load_all_spectra(min_spectra)
    n_classes = len(class_names)

    print(f"Augmenting ({variants_per_class} variants per real spectrum) ...")
    rng = np.random.default_rng(7)
    X_list, y_list = [], []
    for i, name in enumerate(class_names):
        for spec in groups[name]:
            variants = augment(spec, n_variants=variants_per_class, rng=rng)
            X_list.append(variants)
            y_list.extend([i] * variants_per_class)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_classes}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Dataset: {X.shape}  RAM: {X.nbytes/1e9:.2f} GB")

    perm  = np.random.default_rng(0).permutation(len(X))
    X, y  = X[perm], y[perm]
    split = int(0.9 * len(X))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    print(f"Train: {len(X_tr):,}   Val: {len(X_val):,}")

    # ExtraTreesClassifier: random splits instead of best-split search →
    # much faster per tree, lower memory, comparable accuracy.
    # n_jobs=1 avoids Windows threading OOM entirely.
    pipeline = Pipeline([
        ("pca", PCA(n_components=n_components, random_state=42, whiten=True)),
        ("rf",  ExtraTreesClassifier(
                    n_estimators=n_trees,
                    max_depth=20,           # cap depth → bounded memory per tree
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=42,
                    verbose=0,
                )),
    ])

    print(f"\nFitting PCA ({n_components} components) + ExtraTrees ({n_trees} trees, single-threaded) ...")
    t0 = time.time()
    pipeline.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    val_preds = pipeline.predict(X_val)
    val_acc   = accuracy_score(y_val, val_preds)
    print(f"Val accuracy: {val_acc:.4%}")

    # Top-5 accuracy
    probs     = pipeline.predict_proba(X_val)
    top5      = np.argsort(probs, axis=1)[:, -5:]
    top5_hits = sum(y_val[i] in top5[i] for i in range(len(y_val)))
    print(f"Top-5 val accuracy: {top5_hits/len(y_val):.4%}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, config.RF_MODEL_PATH, compress=3)

    # Save / update label encoder (shared with CNN)
    with open(config.LABEL_ENCODER_PATH, "w") as f:
        json.dump({"classes": class_names}, f, indent=2)

    print(f"\nModel saved : {config.RF_MODEL_PATH}")
    print(f"Labels saved: {config.LABEL_ENCODER_PATH}")

    import core.rf_model as rf_mod
    rf_mod.reload()
    print("RF model reloaded — restart Flask to use it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees",       type=int, default=300)
    ap.add_argument("--components",  type=int, default=120)
    ap.add_argument("--variants",    type=int, default=15)
    ap.add_argument("--min_spectra", type=int, default=2)
    args = ap.parse_args()
    train(n_trees=args.trees, n_components=args.components,
          variants_per_class=args.variants, min_spectra=args.min_spectra)

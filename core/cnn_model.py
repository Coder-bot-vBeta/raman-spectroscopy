"""
1D ResNet classifier for Raman spectra — PyTorch backend.
CNN_AVAILABLE is False when model files are absent or unreliable.
"""
import os, json
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CNN_AVAILABLE = False
_model  = None
_labels = None
_device = None

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ── Architecture: 1D ResNet ───────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """Pre-activation residual block for 1-D signals."""
    def __init__(self, channels: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class RamanResNet(nn.Module):
    def __init__(self, input_size: int, n_classes: int,
                 base_ch: int = 32, n_blocks: int = 6, dropout: float = 0.2):
        super().__init__()
        # Stem: encode raw spectrum into feature maps
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(base_ch), nn.GELU(),
            nn.Conv1d(base_ch, base_ch, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch), nn.GELU(),
        )
        # Residual blocks at base resolution
        self.stage1 = nn.Sequential(*[ResBlock1D(base_ch, 7, dropout) for _ in range(n_blocks // 2)])

        # Downsample + widen
        ch2 = base_ch * 2
        self.down = nn.Sequential(
            nn.Conv1d(base_ch, ch2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(ch2), nn.GELU(),
        )
        self.stage2 = nn.Sequential(*[ResBlock1D(ch2, 5, dropout) for _ in range(n_blocks // 2)])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch2, 512), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):          # x: (B, L)
        x = x.unsqueeze(1)         # (B, 1, L)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x = self.pool(x)
        return self.head(x)


def build_model(input_size: int, n_classes: int) -> "RamanResNet":
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not installed.")
    return RamanResNet(input_size, n_classes)


# ── Load ──────────────────────────────────────────────────────────────────────
def _try_load():
    global CNN_AVAILABLE, _model, _labels, _device
    if not _TORCH_OK:
        return
    if not (os.path.exists(config.CNN_MODEL_PATH) and
            os.path.exists(config.LABEL_ENCODER_PATH)):
        return
    try:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(config.LABEL_ENCODER_PATH) as f:
            _labels = json.load(f)["classes"]
        ckpt = torch.load(config.CNN_MODEL_PATH, map_location=_device, weights_only=False)
        n_classes = len(_labels)
        net = RamanResNet(config.GRID_POINTS, n_classes).to(_device)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        _model = net
        CNN_AVAILABLE = True
        print(f"[CNN] Loaded -- {n_classes} classes, device={_device}")
    except Exception as e:
        print(f"[CNN] Load failed: {e}")

_try_load()


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(query: np.ndarray, top_k: int = None) -> list:
    if not CNN_AVAILABLE:
        raise RuntimeError("CNN not available. Run: python training/train_cnn.py")
    import torch, torch.nn.functional as F
    top_k = top_k or config.TOP_K
    x = torch.tensor(query, dtype=torch.float32).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(x)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    from database.reference_store import get_database
    _, _, metadata = get_database()
    results = []
    for idx in top_idx:
        mineral = _labels[idx]
        prob    = float(probs[idx])
        meta    = metadata.get(mineral, {})
        results.append({
            "mineral":        mineral,
            "formula":        meta.get("formula", ""),
            "similarity":     round(prob, 4),
            "confidence_pct": round(prob * 100, 1),
            "source":         "cnn",
            "color":          meta.get("color", "#7c5c3e"),
            "description":    meta.get("description", ""),
            "peaks":          meta.get("peaks", []),
        })
    return results


def reload():
    global CNN_AVAILABLE, _model, _labels, _device
    CNN_AVAILABLE = False
    _model = _labels = _device = None
    _try_load()

import sys, json, torch
sys.path.insert(0, '.')
import config
from core.cnn_model import RamanResNet, _try_load, CNN_AVAILABLE

print(f"CNN_AVAILABLE at import: {CNN_AVAILABLE}")

# Try loading manually with full error
try:
    ckpt   = torch.load('models/cnn_mineral.pt', map_location='cpu', weights_only=False)
    labels = json.load(open('models/label_encoder.json'))['classes']
    print(f"Classes   : {len(labels)}")
    print(f"Val acc   : {ckpt.get('val_acc', 'N/A')}")
    print(f"Arch      : {ckpt.get('arch', 'unknown')}")
    net = RamanResNet(config.GRID_POINTS, len(labels))
    net.load_state_dict(ckpt['model_state'])
    params = sum(p.numel() for p in net.parameters())
    print(f"Params    : {params:,}")
    print("State dict: OK")
except Exception as e:
    import traceback; traceback.print_exc()

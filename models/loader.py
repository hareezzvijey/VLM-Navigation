"""Model loading with CPU fallback handling"""
import sys
import io
import torch
from pathlib import Path

_CPU_FALLBACK = False


class _StderrCapture(io.StringIO):
    def __init__(self, real):
        super().__init__()
        self._real = real

    def write(self, s):
        self._real.write(s)
        super().write(s)

    def flush(self):
        self._real.flush()


def load_models(dino_config: str, dino_ckpt: str, sam_ckpt: str, device: torch.device):
    """Load GroundingDINO and SAM models with CPU fallback detection."""
    global _CPU_FALLBACK

    _cap = _StderrCapture(sys.stderr)
    sys.stderr = _cap

    try:
        from segment_anything import sam_model_registry, SamPredictor
        from groundingdino.util.inference import load_model, predict
    finally:
        sys.stderr = _cap._real

    if "Failed to load custom C++ ops" in _cap.getvalue():
        _CPU_FALLBACK = True
        print("[Pipeline] WARNING: CPU fallback -- thresholds auto-lowered")

    print("[Pipeline] Loading Grounding DINO...")
    dino_model = load_model(dino_config, dino_ckpt, device=device)

    print("[Pipeline] Loading SAM...")
    sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt)
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    return dino_model, sam_predictor, _CPU_FALLBACK


def get_cpu_fallback() -> bool:
    return _CPU_FALLBACK
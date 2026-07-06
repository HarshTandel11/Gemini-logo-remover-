"""
Configuration module for the video logo remover backend.

Centralises all constants, paths, model settings, and pipeline parameters
so that every other module imports from a single source of truth.
"""

import logging

import torch
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
MODEL_DIR: Path = BASE_DIR / "models"
TEMP_DIR: Path = BASE_DIR / "temp"
UPLOAD_DIR: Path = BASE_DIR / "uploads"
OUTPUT_DIR: Path = BASE_DIR / "outputs"

for _d in [MODEL_DIR, TEMP_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    _d.mkdir(exist_ok=True)

# ── Video constraints ─────────────────────────────────────────────────────────
MAX_VIDEO_SIZE_MB: int = 500
SUPPORTED_FORMATS: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

# ── Encoding defaults ────────────────────────────────────────────────────────
DEFAULT_OUTPUT_CRF: int = 18
DEFAULT_OUTPUT_CODEC: str = "libx264"

# ── Device selection ──────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

logger.info("Device: %s  |  dtype: %s", DEVICE, DTYPE)

# ── SAM2 ──────────────────────────────────────────────────────────────────────
SAM2_MODEL_CFG: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT: str = str(MODEL_DIR / "sam2.1_hiera_small.pt")

# ── LaMa ──────────────────────────────────────────────────────────────────────
LAMA_MODEL: str = "big-lama"

# ── ProPainter ────────────────────────────────────────────────────────────────
PROPAINTER_DIR: Path = BASE_DIR / "third_party" / "ProPainter"

# ── DiffuEraser ───────────────────────────────────────────────────────────────
DIFFUERASER_DIR: Path = BASE_DIR / "third_party" / "DiffuEraser"

# ── Pipeline parameters ──────────────────────────────────────────────────────
MASK_DILATION_PX: int = 8
LAPLACIAN_LEVELS: int = 5
FRESCO_TEMPORAL_WEIGHT: float = 0.7
SDXL_ENABLED: bool = False
SDXL_CONFIDENCE_THRESHOLD: float = 0.85
SCENE_THRESHOLD: float = 27.0

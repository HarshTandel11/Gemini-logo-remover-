"""
Stage 5a: LaMa Inpainting (Static Logo Removal)

Uses the Simple LaMa inpainting model for fast, high-quality removal of
static logos where the mask remains mostly consistent across frames.
Each frame is processed individually since the mask is near-identical.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class LaMaInpainter:
    """LaMa-based inpainting for static logo removal.

    Leverages the Simple LaMa model to inpaint masked regions in individual
    frames. Best suited for static logos where the mask is consistent across
    the video, allowing independent per-frame processing.
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None

    @property
    def is_loaded(self) -> bool:
        """Check if the LaMa model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the SimpleLama inpainting model.

        Raises:
            ImportError: If simple_lama_inpainting is not installed.
            RuntimeError: If model loading fails.
        """
        if self._model is not None:
            logger.debug("LaMa model already loaded, skipping.")
            return

        logger.info("Loading SimpleLama inpainting model...")
        try:
            from simple_lama_inpainting import SimpleLama

            self._model = SimpleLama()
            logger.info("SimpleLama model loaded successfully.")
        except ImportError:
            logger.error(
                "simple_lama_inpainting package not found. "
                "Install with: pip install simple-lama-inpainting"
            )
            raise
        except Exception as e:
            logger.error("Failed to load SimpleLama model: %s", e)
            raise RuntimeError(f"LaMa model loading failed: {e}") from e

    def inpaint_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint a single frame using the LaMa model.

        Args:
            frame: Input frame as a BGR numpy array (H, W, 3), uint8.
            mask: Binary mask where 255 = region to inpaint, 0 = keep.
                  Shape (H, W) or (H, W, 1), uint8.

        Returns:
            Inpainted frame as a BGR numpy array (H, W, 3), uint8.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If frame/mask dimensions are incompatible.
        """
        if self._model is None:
            raise RuntimeError("LaMa model not loaded. Call load() first.")

        # Validate inputs
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected BGR frame with shape (H, W, 3), got {frame.shape}"
            )

        # Normalize mask to 2D binary
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != frame.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape[:2]} does not match "
                f"frame shape {frame.shape[:2]}"
            )

        # Convert BGR numpy array to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Convert mask to PIL Image (grayscale)
        mask_pil = Image.fromarray(mask, mode="L")

        # Run LaMa inpainting
        try:
            result_pil = self._model(frame_pil, mask_pil)
        except Exception as e:
            logger.error("LaMa inpainting failed: %s", e)
            raise RuntimeError(f"Inpainting failed: {e}") from e

        # Convert result back to BGR numpy array
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr

    def inpaint_batch(
        self, frames: List[np.ndarray], masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Inpaint a batch of frames using the LaMa model.

        For static logos where the mask is mostly the same across frames,
        each frame is processed individually. This is efficient because
        LaMa is fast per-frame and no temporal coherence modeling is needed
        at this stage.

        Args:
            frames: List of BGR numpy arrays (H, W, 3), uint8.
            masks: List of binary masks. Can be a single mask applied to all
                   frames, or one mask per frame.

        Returns:
            List of inpainted BGR numpy arrays.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If frames and masks lists have incompatible lengths.
        """
        if self._model is None:
            raise RuntimeError("LaMa model not loaded. Call load() first.")

        if not frames:
            logger.warning("Empty frames list provided to inpaint_batch.")
            return []

        # Support single mask for all frames (static logo case)
        single_mask = len(masks) == 1
        if not single_mask and len(masks) != len(frames):
            raise ValueError(
                f"Expected 1 or {len(frames)} masks, got {len(masks)}"
            )

        total = len(frames)
        logger.info("Starting LaMa batch inpainting for %d frames...", total)

        results: List[np.ndarray] = []
        for i, frame in enumerate(frames):
            mask = masks[0] if single_mask else masks[i]

            # Skip inpainting if mask is empty (no logo region)
            if np.max(mask) == 0:
                results.append(frame.copy())
                continue

            result = self.inpaint_frame(frame, mask)
            results.append(result)

            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(
                    "LaMa inpainting progress: %d/%d frames", i + 1, total
                )

        logger.info("LaMa batch inpainting complete: %d frames processed.", total)
        return results

    def unload(self) -> None:
        """Unload the LaMa model and free GPU memory."""
        if self._model is not None:
            logger.info("Unloading LaMa model...")
            del self._model
            self._model = None

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared after LaMa unload.")
            except ImportError:
                pass

            logger.info("LaMa model unloaded.")
        else:
            logger.debug("LaMa model not loaded, nothing to unload.")

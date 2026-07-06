"""
Stage 7: SDXL Inpainting (Optional)

Provides high-quality inpainting using the Stable Diffusion XL inpainting
pipeline as an optional enhancement step. Only activates when inpainting
confidence is below a configurable threshold, allowing selective application
to frames that need additional refinement.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SDXL_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_PROMPT = "clean background, high quality, seamless texture"
DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, artifact, blur, distortion"


class SDXLInpainter:
    """SDXL-based inpainting for optional high-quality refinement.

    Loads the SDXL inpainting pipeline only when explicitly enabled.
    Uses fp16 precision, model CPU offloading, and VAE slicing for
    memory efficiency. Designed to selectively process frames where
    the prior inpainting confidence is below a threshold.
    """

    def __init__(
        self,
        enabled: bool = False,
        model_id: str = DEFAULT_SDXL_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.6,
    ) -> None:
        """Initialize SDXLInpainter.

        Args:
            enabled: Whether SDXL inpainting is enabled.
            model_id: HuggingFace model ID for the SDXL inpainting pipeline.
            confidence_threshold: Confidence threshold below which SDXL is applied.
            prompt: Text prompt for inpainting.
            negative_prompt: Negative prompt to avoid unwanted artifacts.
            num_inference_steps: Number of diffusion steps.
            guidance_scale: Classifier-free guidance scale.
            strength: Denoising strength (0 = no change, 1 = full denoise).
        """
        self._enabled = enabled
        self._model_id = model_id
        self._confidence_threshold = confidence_threshold
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._strength = strength
        self._pipeline: Optional[object] = None

    @property
    def is_loaded(self) -> bool:
        """Check if the SDXL pipeline is currently loaded."""
        return self._pipeline is not None

    @property
    def enabled(self) -> bool:
        """Check if SDXL inpainting is enabled."""
        return self._enabled

    def load(self) -> None:
        """Load the SDXL inpainting pipeline.

        Only loads if SDXL is enabled in configuration. Uses fp16 precision,
        model CPU offloading, and VAE slicing for memory efficiency.

        Raises:
            RuntimeError: If SDXL is enabled but loading fails.
        """
        if not self._enabled:
            logger.info("SDXL inpainting is disabled, skipping load.")
            return

        if self._pipeline is not None:
            logger.debug("SDXL pipeline already loaded, skipping.")
            return

        logger.info("Loading SDXL inpainting pipeline: %s", self._model_id)

        try:
            import torch
            from diffusers import AutoPipelineForInpainting

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            pipeline = AutoPipelineForInpainting.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
                safety_checker=None,
            )
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()

            self._pipeline = pipeline
            logger.info("SDXL inpainting pipeline loaded successfully.")

        except ImportError:
            logger.error(
                "diffusers package not found. "
                "Install with: pip install diffusers transformers accelerate"
            )
            raise
        except Exception as e:
            logger.error("Failed to load SDXL pipeline: %s", e)
            raise RuntimeError(f"SDXL pipeline loading failed: {e}") from e

    def should_apply(self, confidence: float) -> bool:
        """Determine if SDXL inpainting should be applied based on confidence.

        Args:
            confidence: Inpainting confidence score from the previous stage
                        (0.0 = poor quality, 1.0 = excellent quality).

        Returns:
            True if SDXL is enabled and confidence is below threshold.
        """
        if not self._enabled:
            return False

        apply = confidence < self._confidence_threshold
        if apply:
            logger.debug(
                "SDXL will be applied: confidence %.3f < threshold %.3f",
                confidence,
                self._confidence_threshold,
            )
        else:
            logger.debug(
                "SDXL skipped: confidence %.3f >= threshold %.3f",
                confidence,
                self._confidence_threshold,
            )
        return apply

    def inpaint_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> np.ndarray:
        """Inpaint a single frame using the SDXL pipeline.

        Args:
            frame: Input frame as a BGR numpy array (H, W, 3), uint8.
            mask: Binary mask where 255 = region to inpaint, 0 = keep.
                  Shape (H, W) or (H, W, 1), uint8.
            prompt: Optional override for the default prompt.
            negative_prompt: Optional override for the default negative prompt.

        Returns:
            Inpainted frame as a BGR numpy array (H, W, 3), uint8.

        Raises:
            RuntimeError: If the pipeline is not loaded or SDXL is disabled.
        """
        if not self._enabled:
            raise RuntimeError("SDXL inpainting is disabled.")
        if self._pipeline is None:
            raise RuntimeError("SDXL pipeline not loaded. Call load() first.")

        # Normalize mask to 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        original_h, original_w = frame.shape[:2]

        # Convert to PIL Images
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        mask_pil = Image.fromarray(mask, mode="L")

        # SDXL requires dimensions divisible by 8
        proc_h = (original_h // 8) * 8
        proc_w = (original_w // 8) * 8

        # Clamp to reasonable SDXL dimensions (max 1024 for SDXL)
        max_dim = 1024
        if proc_h > max_dim or proc_w > max_dim:
            scale = max_dim / max(proc_h, proc_w)
            proc_h = int(proc_h * scale) // 8 * 8
            proc_w = int(proc_w * scale) // 8 * 8

        frame_pil = frame_pil.resize((proc_w, proc_h), Image.LANCZOS)
        mask_pil = mask_pil.resize((proc_w, proc_h), Image.NEAREST)

        # Run SDXL inpainting
        try:
            result = self._pipeline(
                prompt=prompt or self._prompt,
                negative_prompt=negative_prompt or self._negative_prompt,
                image=frame_pil,
                mask_image=mask_pil,
                num_inference_steps=self._num_inference_steps,
                guidance_scale=self._guidance_scale,
                strength=self._strength,
            )
            result_pil = result.images[0]
        except Exception as e:
            logger.error("SDXL inpainting failed: %s", e)
            raise RuntimeError(f"SDXL inpainting failed: {e}") from e

        # Resize back to original dimensions
        if result_pil.size != (original_w, original_h):
            result_pil = result_pil.resize(
                (original_w, original_h), Image.LANCZOS
            )

        # Convert back to BGR numpy
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr

    def unload(self) -> None:
        """Unload the SDXL pipeline and free GPU memory."""
        if self._pipeline is not None:
            logger.info("Unloading SDXL inpainting pipeline...")
            del self._pipeline
            self._pipeline = None

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared after SDXL unload.")
            except ImportError:
                pass

            logger.info("SDXL pipeline unloaded.")
        else:
            logger.debug("SDXL pipeline not loaded, nothing to unload.")

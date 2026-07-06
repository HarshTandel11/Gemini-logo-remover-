"""
Stage 6: DiffuEraser Refinement

Applies diffusion-based refinement to inpainted frames using DiffuEraser
or a Stable Diffusion 1.5 inpainting fallback. Also provides a confidence
scoring mechanism to evaluate inpainting quality based on boundary
smoothness, edge continuity, and structural similarity.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default prompt for refinement
DEFAULT_PROMPT = "clean background, seamless"

# Confidence scoring weights
_WEIGHT_HISTOGRAM = 0.35
_WEIGHT_EDGE = 0.30
_WEIGHT_SSIM = 0.35


class DiffuEraserRefiner:
    """Diffusion-based refinement for inpainted frames.

    Attempts to load DiffuEraser from HuggingFace for specialized inpainting
    refinement. Falls back to the RunwayML Stable Diffusion 1.5 inpainting
    pipeline if DiffuEraser is unavailable. Both use fp16 precision and
    model CPU offloading for memory efficiency.
    """

    def __init__(self) -> None:
        self._pipeline: Optional[object] = None
        self._model_name: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Check if the refinement pipeline is currently loaded."""
        return self._pipeline is not None

    @property
    def model_name(self) -> Optional[str]:
        """Return the name of the currently loaded model."""
        return self._model_name

    def load(self) -> None:
        """Load the diffusion inpainting pipeline.

        First attempts to load DiffuEraser from HuggingFace. If unavailable,
        falls back to runwayml/stable-diffusion-inpainting (SD 1.5).
        Both pipelines use float16 precision and model CPU offloading.

        Raises:
            RuntimeError: If neither DiffuEraser nor SD 1.5 can be loaded.
        """
        if self._pipeline is not None:
            logger.debug("Refinement pipeline already loaded, skipping.")
            return

        import torch
        from diffusers import StableDiffusionInpaintPipeline

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Attempt 1: Load DiffuEraser
        try:
            logger.info("Attempting to load DiffuEraser pipeline...")
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "luckyhzt/DiffuEraser",
                torch_dtype=dtype,
                safety_checker=None,
            )
            pipeline.enable_model_cpu_offload()
            self._pipeline = pipeline
            self._model_name = "DiffuEraser"
            logger.info("DiffuEraser pipeline loaded successfully.")
            return
        except Exception as e:
            logger.warning(
                "Failed to load DiffuEraser, falling back to SD 1.5: %s", e
            )

        # Attempt 2: Fall back to Stable Diffusion 1.5 Inpainting
        try:
            logger.info("Loading SD 1.5 inpainting pipeline (fallback)...")
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=dtype,
                safety_checker=None,
            )
            pipeline.enable_model_cpu_offload()
            self._pipeline = pipeline
            self._model_name = "SD-1.5-Inpainting"
            logger.info("SD 1.5 inpainting pipeline loaded successfully.")
        except Exception as e:
            logger.error("Failed to load any inpainting pipeline: %s", e)
            raise RuntimeError(
                "Could not load DiffuEraser or SD 1.5 inpainting. "
                "Ensure diffusers is installed and models are accessible."
            ) from e

    def refine_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        prompt: str = DEFAULT_PROMPT,
    ) -> np.ndarray:
        """Refine a single inpainted frame using diffusion.

        Args:
            frame: Input frame as a BGR numpy array (H, W, 3), uint8.
            mask: Binary mask where 255 = region to refine, 0 = keep.
                  Shape (H, W) or (H, W, 1), uint8.
            prompt: Text prompt guiding the refinement.

        Returns:
            Refined frame as a BGR numpy array (H, W, 3), uint8.

        Raises:
            RuntimeError: If the pipeline is not loaded.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        # Normalize mask to 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        original_h, original_w = frame.shape[:2]

        # Convert to PIL Images (RGB for frame, L for mask)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        mask_pil = Image.fromarray(mask, mode="L")

        # SD pipelines require dimensions divisible by 8
        proc_h = (original_h // 8) * 8
        proc_w = (original_w // 8) * 8
        if proc_h != original_h or proc_w != original_w:
            frame_pil = frame_pil.resize((proc_w, proc_h), Image.LANCZOS)
            mask_pil = mask_pil.resize((proc_w, proc_h), Image.NEAREST)

        # Run diffusion inpainting
        try:
            result = self._pipeline(
                prompt=prompt,
                image=frame_pil,
                mask_image=mask_pil,
                num_inference_steps=20,
                guidance_scale=7.5,
            )
            result_pil = result.images[0]
        except Exception as e:
            logger.error("Diffusion refinement failed: %s", e)
            raise RuntimeError(f"Refinement failed: {e}") from e

        # Resize back to original dimensions if needed
        if proc_h != original_h or proc_w != original_w:
            result_pil = result_pil.resize(
                (original_w, original_h), Image.LANCZOS
            )

        # Convert back to BGR numpy
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr

    def refine_batch(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        prompt: str = DEFAULT_PROMPT,
    ) -> List[np.ndarray]:
        """Refine a batch of inpainted frames.

        Args:
            frames: List of BGR numpy arrays (H, W, 3), uint8.
            masks: List of binary masks. Can be a single mask for all frames
                   or one mask per frame.
            prompt: Text prompt guiding the refinement.

        Returns:
            List of refined BGR numpy arrays.

        Raises:
            RuntimeError: If the pipeline is not loaded.
            ValueError: If frames and masks have incompatible lengths.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        if not frames:
            logger.warning("Empty frames list provided to refine_batch.")
            return []

        single_mask = len(masks) == 1
        if not single_mask and len(masks) != len(frames):
            raise ValueError(
                f"Expected 1 or {len(frames)} masks, got {len(masks)}"
            )

        total = len(frames)
        logger.info("Starting diffusion refinement for %d frames...", total)

        results: List[np.ndarray] = []
        for i, frame in enumerate(frames):
            mask = masks[0] if single_mask else masks[i]

            # Skip refinement if mask is empty
            if np.max(mask) == 0:
                results.append(frame.copy())
                continue

            result = self.refine_frame(frame, mask, prompt)
            results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(
                    "Diffusion refinement progress: %d/%d frames", i + 1, total
                )

        logger.info("Diffusion refinement complete: %d frames processed.", total)
        return results

    @staticmethod
    def compute_confidence(
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute a confidence score for the inpainting quality.

        Evaluates quality using three metrics:
        1. Color histogram similarity at the mask boundary
        2. Edge continuity across the mask boundary
        3. Structural similarity (SSIM) in the boundary region

        Args:
            original: Original frame (BGR, uint8).
            inpainted: Inpainted frame (BGR, uint8).
            mask: Binary mask (255 = inpainted region), (H, W), uint8.

        Returns:
            Confidence score between 0.0 (poor) and 1.0 (excellent).
        """
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Create boundary region (dilated mask - eroded mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = cv2.subtract(dilated, eroded)

        # If boundary is empty, return high confidence
        if np.sum(boundary) == 0:
            return 1.0

        # --- Metric 1: Color Histogram Similarity at Boundary ---
        hist_score = _compute_histogram_similarity(
            original, inpainted, boundary
        )

        # --- Metric 2: Edge Continuity ---
        edge_score = _compute_edge_continuity(inpainted, mask, boundary)

        # --- Metric 3: SSIM in Boundary Region ---
        ssim_score = _compute_boundary_ssim(original, inpainted, boundary)

        # Weighted combination
        confidence = (
            _WEIGHT_HISTOGRAM * hist_score
            + _WEIGHT_EDGE * edge_score
            + _WEIGHT_SSIM * ssim_score
        )

        confidence = float(np.clip(confidence, 0.0, 1.0))

        logger.debug(
            "Confidence: %.3f (hist=%.3f, edge=%.3f, ssim=%.3f)",
            confidence,
            hist_score,
            edge_score,
            ssim_score,
        )

        return confidence

    def unload(self) -> None:
        """Unload the diffusion pipeline and free GPU memory."""
        if self._pipeline is not None:
            logger.info("Unloading %s pipeline...", self._model_name)
            del self._pipeline
            self._pipeline = None
            self._model_name = None

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared after pipeline unload.")
            except ImportError:
                pass

            logger.info("Refinement pipeline unloaded.")
        else:
            logger.debug("Pipeline not loaded, nothing to unload.")


# ---------------------------------------------------------------------------
# Private helper functions for confidence scoring
# ---------------------------------------------------------------------------


def _compute_histogram_similarity(
    original: np.ndarray,
    inpainted: np.ndarray,
    boundary: np.ndarray,
) -> float:
    """Compare color histograms of original and inpainted boundary regions.

    Args:
        original: Original frame (BGR, uint8).
        inpainted: Inpainted frame (BGR, uint8).
        boundary: Binary boundary mask, uint8.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    scores = []
    for ch in range(3):
        hist_orig = cv2.calcHist(
            [original], [ch], boundary, [64], [0, 256]
        )
        hist_inp = cv2.calcHist(
            [inpainted], [ch], boundary, [64], [0, 256]
        )
        cv2.normalize(hist_orig, hist_orig)
        cv2.normalize(hist_inp, hist_inp)
        score = cv2.compareHist(
            hist_orig, hist_inp, cv2.HISTCMP_CORREL
        )
        scores.append(max(score, 0.0))

    return float(np.mean(scores))


def _compute_edge_continuity(
    inpainted: np.ndarray,
    mask: np.ndarray,
    boundary: np.ndarray,
) -> float:
    """Measure edge continuity across the inpainting boundary.

    Computes Canny edges on the inpainted result and checks for
    discontinuities at the boundary. Fewer boundary edges = smoother
    transition = higher score.

    Args:
        inpainted: Inpainted frame (BGR, uint8).
        mask: Original binary mask, uint8.
        boundary: Binary boundary mask, uint8.

    Returns:
        Edge continuity score between 0.0 and 1.0.
    """
    gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    boundary_pixels = np.sum(boundary > 0)
    if boundary_pixels == 0:
        return 1.0

    # Count edge pixels in boundary region
    boundary_edges = np.sum((edges > 0) & (boundary > 0))
    edge_density = boundary_edges / boundary_pixels

    # Lower edge density at boundary = better continuity
    # Map [0, 0.5] -> [1.0, 0.0]
    score = max(0.0, 1.0 - (edge_density / 0.5))
    return float(score)


def _compute_boundary_ssim(
    original: np.ndarray,
    inpainted: np.ndarray,
    boundary: np.ndarray,
) -> float:
    """Compute structural similarity in the boundary region.

    Uses a simplified SSIM computation focused on the boundary area
    between original and inpainted frames.

    Args:
        original: Original frame (BGR, uint8).
        inpainted: Inpainted frame (BGR, uint8).
        boundary: Binary boundary mask, uint8.

    Returns:
        SSIM score between 0.0 and 1.0.
    """
    # Convert to grayscale floats
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_inp = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY).astype(np.float64)
    boundary_f = (boundary > 0).astype(np.float64)

    # SSIM constants
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Compute local statistics with Gaussian blur
    ksize = 11
    mu_orig = cv2.GaussianBlur(gray_orig, (ksize, ksize), 1.5)
    mu_inp = cv2.GaussianBlur(gray_inp, (ksize, ksize), 1.5)

    mu_orig_sq = mu_orig ** 2
    mu_inp_sq = mu_inp ** 2
    mu_cross = mu_orig * mu_inp

    sigma_orig_sq = cv2.GaussianBlur(
        gray_orig ** 2, (ksize, ksize), 1.5
    ) - mu_orig_sq
    sigma_inp_sq = cv2.GaussianBlur(
        gray_inp ** 2, (ksize, ksize), 1.5
    ) - mu_inp_sq
    sigma_cross = cv2.GaussianBlur(
        gray_orig * gray_inp, (ksize, ksize), 1.5
    ) - mu_cross

    # SSIM map
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_orig_sq + mu_inp_sq + c1) * (sigma_orig_sq + sigma_inp_sq + c2)
    ssim_map = numerator / (denominator + 1e-10)

    # Average SSIM only in boundary region
    boundary_sum = np.sum(boundary_f)
    if boundary_sum == 0:
        return 1.0

    ssim_val = np.sum(ssim_map * boundary_f) / boundary_sum
    return float(np.clip(ssim_val, 0.0, 1.0))

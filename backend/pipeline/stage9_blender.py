"""
Stage 9: Multi-band Laplacian Blending

Performs seamless composition of inpainted and original frames using
multi-band Laplacian pyramid blending. This ensures smooth, artifact-free
transitions at the mask boundary by blending at multiple frequency bands.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default number of pyramid levels
DEFAULT_NUM_LEVELS = 6


class LaplacianBlender:
    """Multi-band Laplacian pyramid blender.

    Blends inpainted and original frames using Gaussian and Laplacian
    pyramids, producing seamless transitions at mask boundaries. Each
    frequency band is blended independently using a Gaussian-smoothed
    mask pyramid, then the result is reconstructed from the blended
    Laplacian pyramid.
    """

    def __init__(self, num_levels: int = DEFAULT_NUM_LEVELS) -> None:
        """Initialize the Laplacian blender.

        Args:
            num_levels: Number of pyramid levels for blending.
                        Higher values produce smoother blends but are slower.
        """
        self._num_levels = num_levels

    def blend(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Blend an inpainted frame with the original using Laplacian pyramids.

        The mask determines which regions come from the inpainted frame
        (255) and which from the original (0). The blending happens across
        multiple frequency bands for a seamless result.

        Args:
            inpainted: Inpainted frame (BGR, uint8), (H, W, 3).
            original: Original frame (BGR, uint8), (H, W, 3).
            mask: Binary mask (uint8), (H, W) or (H, W, 1).
                  255 = use inpainted, 0 = use original.

        Returns:
            Blended frame (BGR, uint8), (H, W, 3).
        """
        # Normalize mask to 2D float [0, 1]
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_f = mask.astype(np.float32) / 255.0

        # Convert to 3-channel mask for per-channel blending
        mask_3ch = np.stack([mask_f] * 3, axis=-1)

        # Ensure inputs are float32 for pyramid operations
        inp_f = inpainted.astype(np.float32)
        orig_f = original.astype(np.float32)

        # Compute max feasible pyramid levels based on image dimensions
        h, w = inpainted.shape[:2]
        max_levels = self._max_pyramid_levels(h, w)
        num_levels = min(self._num_levels, max_levels)

        if num_levels < 2:
            # Fallback to simple alpha blending if image is too small
            logger.debug(
                "Image too small for pyramid blending (%dx%d), "
                "falling back to alpha blend.",
                w,
                h,
            )
            blended = orig_f * (1.0 - mask_3ch) + inp_f * mask_3ch
            return np.clip(blended, 0, 255).astype(np.uint8)

        # Build Laplacian pyramids for both images
        lap_inpainted = self._build_laplacian_pyramid(inp_f, num_levels)
        lap_original = self._build_laplacian_pyramid(orig_f, num_levels)

        # Build Gaussian pyramid for the mask
        gauss_mask = self._build_gaussian_pyramid(mask_3ch, num_levels)

        # Blend Laplacian pyramids at each level
        lap_blended: List[np.ndarray] = []
        for i in range(num_levels):
            m = gauss_mask[i]
            blended_level = lap_original[i] * (1.0 - m) + lap_inpainted[i] * m
            lap_blended.append(blended_level)

        # Reconstruct from blended Laplacian pyramid
        result = self._reconstruct(lap_blended)

        return np.clip(result, 0, 255).astype(np.uint8)

    def blend_batch(
        self,
        inpainted_frames: List[np.ndarray],
        original_frames: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Blend a batch of inpainted frames with originals.

        Args:
            inpainted_frames: List of inpainted BGR frames.
            original_frames: List of original BGR frames.
            masks: List of binary masks. Can be a single mask for all frames
                   or one mask per frame.

        Returns:
            List of blended BGR frames.

        Raises:
            ValueError: If input list lengths are incompatible.
        """
        n = len(inpainted_frames)
        if len(original_frames) != n:
            raise ValueError(
                f"Mismatched frame counts: {n} inpainted vs "
                f"{len(original_frames)} original."
            )

        single_mask = len(masks) == 1
        if not single_mask and len(masks) != n:
            raise ValueError(
                f"Expected 1 or {n} masks, got {len(masks)}"
            )

        if n == 0:
            return []

        logger.info("Starting Laplacian blending for %d frames...", n)

        results: List[np.ndarray] = []
        for i in range(n):
            mask = masks[0] if single_mask else masks[i]
            blended = self.blend(inpainted_frames[i], original_frames[i], mask)
            results.append(blended)

            if (i + 1) % 100 == 0 or (i + 1) == n:
                logger.info("Blending progress: %d/%d frames", i + 1, n)

        logger.info("Laplacian blending complete: %d frames.", n)
        return results

    @staticmethod
    def _build_gaussian_pyramid(
        image: np.ndarray, num_levels: int
    ) -> List[np.ndarray]:
        """Build a Gaussian pyramid for an image.

        Args:
            image: Input image (float32), (H, W, C) or (H, W).
            num_levels: Number of pyramid levels.

        Returns:
            List of images from finest (original) to coarsest resolution.
        """
        pyramid = [image.copy()]
        current = image.copy()

        for _ in range(num_levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)

        return pyramid

    @staticmethod
    def _build_laplacian_pyramid(
        image: np.ndarray, num_levels: int
    ) -> List[np.ndarray]:
        """Build a Laplacian pyramid for an image.

        Each level contains the detail (high-frequency) information lost
        when downsampling. The final level is the coarsest Gaussian level.

        Args:
            image: Input image (float32), (H, W, C) or (H, W).
            num_levels: Number of pyramid levels.

        Returns:
            List of Laplacian levels from finest to coarsest.
            The last element is the smallest Gaussian level.
        """
        gaussian_pyramid: List[np.ndarray] = [image.copy()]
        current = image.copy()

        for _ in range(num_levels - 1):
            current = cv2.pyrDown(current)
            gaussian_pyramid.append(current)

        laplacian_pyramid: List[np.ndarray] = []
        for i in range(num_levels - 1):
            h, w = gaussian_pyramid[i].shape[:2]
            upsampled = cv2.pyrUp(
                gaussian_pyramid[i + 1], dstsize=(w, h)
            )
            laplacian = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(laplacian)

        # The coarsest level is kept as-is (low-pass residual)
        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    @staticmethod
    def _reconstruct(laplacian_pyramid: List[np.ndarray]) -> np.ndarray:
        """Reconstruct an image from its Laplacian pyramid.

        Args:
            laplacian_pyramid: Laplacian pyramid from coarsest to finest.
                               (list order: finest first, coarsest last)

        Returns:
            Reconstructed image (float32).
        """
        # Start from the coarsest level
        current = laplacian_pyramid[-1].copy()

        # Traverse from coarsest to finest
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            h, w = laplacian_pyramid[i].shape[:2]
            upsampled = cv2.pyrUp(current, dstsize=(w, h))
            current = upsampled + laplacian_pyramid[i]

        return current

    @staticmethod
    def _max_pyramid_levels(h: int, w: int) -> int:
        """Compute the maximum number of feasible pyramid levels.

        Each pyrDown halves dimensions; we stop when either dimension
        would fall below 4 pixels.

        Args:
            h: Image height.
            w: Image width.

        Returns:
            Maximum number of pyramid levels.
        """
        min_dim = min(h, w)
        levels = 1
        while min_dim >= 8:
            min_dim //= 2
            levels += 1
        return levels

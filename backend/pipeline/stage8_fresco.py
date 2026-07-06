"""
Stage 8: FRESCO Temporal Enhancement

Applies temporal consistency enhancement to inpainted frames using
optical-flow-based warping and temporal Gaussian filtering. Ensures
smooth transitions between frames in the inpainted (masked) regions
by blending warped neighboring frames and filtering along the time axis.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FRESCOEnhancer:
    """FRESCO-inspired temporal consistency enhancer.

    Improves temporal coherence of inpainted video frames by:
    1. Warping neighboring frames using optical flow into each target frame
    2. Blending warped frames in the masked (inpainted) regions
    3. Applying a temporal Gaussian filter along the time axis for smooth
       transitions across the video sequence.
    """

    def enhance(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        flows_forward: List[Optional[np.ndarray]],
        blend_weight: float = 0.3,
        temporal_kernel_size: int = 5,
    ) -> List[np.ndarray]:
        """Apply temporal consistency enhancement to inpainted frames.

        For each frame, warps the previous and next frames using optical flow,
        blends them into the masked region, and then applies a temporal
        Gaussian filter across the sequence for smooth transitions.

        Args:
            frames: List of inpainted BGR frames (H, W, 3), uint8.
            masks: List of binary masks (H, W), uint8, where 255 = inpainted.
                   Can be a single mask applied to all frames.
            flows_forward: List of forward optical flow arrays (H, W, 2), float32.
                           flows_forward[i] maps frame[i] -> frame[i+1].
                           Last element should be None. Length must be len(frames).
            blend_weight: Weight for the warped neighbor contribution in
                          masked regions (default 0.3).
            temporal_kernel_size: Kernel size for the temporal Gaussian filter
                                  (default 5, must be odd).

        Returns:
            List of temporally enhanced BGR frames.

        Raises:
            ValueError: If input lengths are inconsistent.
        """
        n = len(frames)
        if n == 0:
            logger.warning("Empty frames list, returning empty.")
            return []

        if len(flows_forward) != n:
            raise ValueError(
                f"Expected {n} flow entries, got {len(flows_forward)}"
            )

        # Support single mask for all frames
        single_mask = len(masks) == 1
        if not single_mask and len(masks) != n:
            raise ValueError(
                f"Expected 1 or {n} masks, got {len(masks)}"
            )

        logger.info(
            "Starting FRESCO temporal enhancement on %d frames "
            "(blend_weight=%.2f, kernel=%d)...",
            n,
            blend_weight,
            temporal_kernel_size,
        )

        # --- Step 1: Flow-based neighbor blending ---
        blended = []
        for i in range(n):
            mask = masks[0] if single_mask else masks[i]
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            frame = frames[i].copy()

            # If mask is empty, skip blending
            if np.max(mask) == 0:
                blended.append(frame)
                continue

            mask_f = (mask > 127).astype(np.float32)
            accumulator = frame.astype(np.float32)
            weight_sum = np.ones_like(mask_f)

            # Warp and blend previous frame
            if i > 0 and flows_forward[i - 1] is not None:
                # Forward flow from frame[i-1] -> frame[i]
                warped_prev = self._warp_frame(frames[i - 1], flows_forward[i - 1])
                accumulator += warped_prev.astype(np.float32) * blend_weight * mask_f[:, :, np.newaxis]
                weight_sum += blend_weight * mask_f

            # Warp and blend next frame
            if i < n - 1 and flows_forward[i] is not None:
                # Invert forward flow for backward warping (approximate)
                backward_flow = -flows_forward[i]
                warped_next = self._warp_frame(frames[i + 1], backward_flow)
                accumulator += warped_next.astype(np.float32) * blend_weight * mask_f[:, :, np.newaxis]
                weight_sum += blend_weight * mask_f

            # Normalize by weight sum in masked region
            weight_sum_3ch = weight_sum[:, :, np.newaxis]
            result = accumulator / (weight_sum_3ch + 1e-8)

            # Only apply blending in masked region
            mask_3ch = mask_f[:, :, np.newaxis]
            output = frame.astype(np.float32) * (1.0 - mask_3ch) + result * mask_3ch
            blended.append(np.clip(output, 0, 255).astype(np.uint8))

        # --- Step 2: Temporal Gaussian filtering ---
        enhanced = self._temporal_filter(
            blended, masks, kernel_size=temporal_kernel_size
        )

        logger.info("FRESCO temporal enhancement complete.")
        return enhanced

    @staticmethod
    def _warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp a frame using optical flow via cv2.remap.

        Args:
            frame: Input BGR frame (H, W, 3), uint8.
            flow: Optical flow field (H, W, 2), float32.
                  flow[y, x] = (dx, dy) displacement at pixel (x, y).

        Returns:
            Warped frame (H, W, 3), uint8.
        """
        h, w = flow.shape[:2]

        # Create the sampling grid: base coordinates + flow displacement
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = grid_x + flow[:, :, 0]
        map_y = grid_y + flow[:, :, 1]

        # Remap using bilinear interpolation
        warped = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return warped

    @staticmethod
    def _temporal_filter(
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        kernel_size: int = 5,
    ) -> List[np.ndarray]:
        """Apply temporal Gaussian filtering along the time axis.

        Smooths pixel values across the time dimension only in the masked
        (inpainted) regions, producing temporally consistent output while
        leaving unmasked regions untouched.

        Args:
            frames: List of BGR frames (H, W, 3), uint8.
            masks: List of binary masks (H, W), uint8.
            kernel_size: Size of the temporal Gaussian kernel (must be odd).

        Returns:
            List of temporally filtered BGR frames.
        """
        from scipy.ndimage import gaussian_filter1d

        n = len(frames)
        if n <= 1:
            return [f.copy() for f in frames]

        single_mask = len(masks) == 1

        # Temporal sigma from kernel size
        sigma = (kernel_size - 1) / 4.0
        if sigma < 0.5:
            sigma = 0.5

        # Stack frames into a 4D array: (T, H, W, C)
        stack = np.stack(frames, axis=0).astype(np.float32)

        # Apply Gaussian filter along time axis (axis=0)
        filtered = gaussian_filter1d(stack, sigma=sigma, axis=0)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        # Merge: use filtered values only in masked regions
        results: List[np.ndarray] = []
        for i in range(n):
            mask = masks[0] if single_mask else masks[i]
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            mask_3ch = (mask > 127).astype(np.uint8)[:, :, np.newaxis]

            # Blend: masked region from filtered, rest from original
            output = frames[i] * (1 - mask_3ch) + filtered[i] * mask_3ch
            results.append(output.astype(np.uint8))

        return results

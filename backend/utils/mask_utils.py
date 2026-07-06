"""
Binary-mask manipulation utilities.

All functions operate on NumPy arrays and use OpenCV for morphological
operations.  Masks are expected as 2-D uint8 arrays where **255** = foreground
and **0** = background, unless otherwise noted.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Morphological operations ─────────────────────────────────────────────────


def dilate_mask(
    mask: np.ndarray,
    kernel_size: int = 8,
    iterations: int = 1,
) -> np.ndarray:
    """Dilate a binary mask to expand foreground regions.

    Args:
        mask: ``(H, W)`` uint8 binary mask (0 / 255).
        kernel_size: Size of the square structuring element.
        iterations: Number of dilation passes.

    Returns:
        Dilated mask of the same shape and dtype.
    """
    mask = _ensure_binary(mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated: np.ndarray = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated


def erode_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Erode a binary mask to shrink foreground regions.

    Args:
        mask: ``(H, W)`` uint8 binary mask.
        kernel_size: Size of the square structuring element.
        iterations: Number of erosion passes.

    Returns:
        Eroded mask of the same shape and dtype.
    """
    mask = _ensure_binary(mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    eroded: np.ndarray = cv2.erode(mask, kernel, iterations=iterations)
    return eroded


# ── Cleaning / smoothing ─────────────────────────────────────────────────────


def clean_mask(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove small connected components from a binary mask.

    Connected components with an area (in pixels) strictly below
    *min_area* are zeroed out.

    Args:
        mask: ``(H, W)`` uint8 binary mask.
        min_area: Minimum component area to keep.

    Returns:
        Cleaned mask with small blobs removed.
    """
    mask = _ensure_binary(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)
    for label_idx in range(1, num_labels):  # skip background (0)
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = 255

    removed = num_labels - 1 - int(np.count_nonzero(np.unique(labels[cleaned > 0])))
    if removed > 0:
        logger.debug("Removed %d small components (min_area=%d)", removed, min_area)

    return cleaned


def smooth_mask(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth mask edges using a Gaussian blur, then re-threshold.

    This produces smoother contour boundaries while keeping the mask
    strictly binary.

    Args:
        mask: ``(H, W)`` uint8 binary mask.
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        Smoothed binary mask (uint8, 0/255).
    """
    mask = _ensure_binary(mask)
    # Kernel size must be odd; derive from sigma.
    ksize = int(2 * round(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigma)
    _, smoothed = cv2.threshold(blurred, 127.0, 255.0, cv2.THRESH_BINARY)
    return smoothed.astype(np.uint8)


# ── Feathered / soft-edge mask ────────────────────────────────────────────────


def create_feathered_mask(
    mask: np.ndarray,
    feather_px: int = 10,
) -> np.ndarray:
    """Create a soft-edge float mask in [0, 1] from a binary mask.

    The mask edge is feathered (blurred) over *feather_px* pixels,
    producing a smooth alpha transition suitable for blending.

    Args:
        mask: ``(H, W)`` uint8 binary mask.
        feather_px: Width of the soft transition band in pixels.

    Returns:
        ``(H, W)`` float32 mask with values in ``[0.0, 1.0]``.
    """
    mask = _ensure_binary(mask)
    ksize = feather_px * 2 + 1  # Ensure odd.
    feathered = cv2.GaussianBlur(
        mask.astype(np.float32) / 255.0,
        (ksize, ksize),
        sigmaX=feather_px / 3.0,
    )
    # Clamp to [0, 1] (GaussianBlur should keep it in range, but be safe).
    feathered = np.clip(feathered, 0.0, 1.0)
    return feathered.astype(np.float32)


# ── Bounding-box extraction ──────────────────────────────────────────────────


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute the axis-aligned bounding box of the foreground region.

    Args:
        mask: ``(H, W)`` uint8 binary mask.

    Returns:
        ``(x, y, w, h)`` tuple.  Returns ``(0, 0, 0, 0)`` if the mask is
        entirely zero.
    """
    mask = _ensure_binary(mask)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, w, h)


# ── Temporal / multi-mask analysis ────────────────────────────────────────────


def masks_are_static(
    masks: Sequence[np.ndarray],
    threshold: float = 0.95,
) -> bool:
    """Determine whether a sequence of masks is essentially static.

    Pairwise IoU is computed between consecutive masks.  If the *minimum*
    IoU exceeds *threshold* the sequence is considered static (i.e. the
    logo / watermark does not move).

    Args:
        masks: Sequence of ``(H, W)`` uint8 binary masks.
        threshold: Minimum IoU to consider the sequence static.

    Returns:
        ``True`` if all consecutive IoUs are ≥ *threshold*.
    """
    if len(masks) < 2:
        return True

    for i in range(len(masks) - 1):
        iou = compute_mask_iou(masks[i], masks[i + 1])
        if iou < threshold:
            logger.debug(
                "Masks %d→%d IoU=%.4f < threshold=%.4f → non-static",
                i, i + 1, iou, threshold,
            )
            return False

    return True


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute the Intersection-over-Union (IoU) of two binary masks.

    Args:
        mask1: ``(H, W)`` uint8 binary mask.
        mask2: ``(H, W)`` uint8 binary mask of the same shape.

    Returns:
        IoU as a float in ``[0.0, 1.0]``.  Returns ``0.0`` when both
        masks are empty.
    """
    m1 = _ensure_binary(mask1).astype(bool)
    m2 = _ensure_binary(mask2).astype(bool)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


# ── Combination helpers ───────────────────────────────────────────────────────


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Merge multiple binary masks into a single mask via logical OR.

    Args:
        masks: List of ``(H, W)`` uint8 binary masks of identical shape.

    Returns:
        Combined ``(H, W)`` uint8 binary mask.

    Raises:
        ValueError: If *masks* is empty.
    """
    if not masks:
        raise ValueError("Cannot combine an empty list of masks.")

    combined = np.zeros_like(masks[0])
    for m in masks:
        combined = cv2.bitwise_or(combined, _ensure_binary(m))
    return combined


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert a binary mask (swap foreground and background).

    Args:
        mask: ``(H, W)`` uint8 binary mask.

    Returns:
        Inverted mask.
    """
    return cv2.bitwise_not(_ensure_binary(mask))


# ── Internal helpers ──────────────────────────────────────────────────────────


def _ensure_binary(mask: np.ndarray) -> np.ndarray:
    """Normalise *mask* to a strict 0/255 uint8 array.

    Handles float [0, 1] masks, bool masks, and multi-channel inputs
    transparently.

    Args:
        mask: Input mask array.

    Returns:
        2-D uint8 array with only 0 and 255 values.
    """
    # Squeeze single-channel dim if present (H, W, 1) → (H, W).
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim == 3:
        # Multi-channel: convert to grayscale.
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Float masks in [0, 1].
    if mask.dtype in (np.float32, np.float64):
        mask = (mask * 255).clip(0, 255).astype(np.uint8)

    # Bool masks.
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    # Threshold to ensure strict binary.
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary

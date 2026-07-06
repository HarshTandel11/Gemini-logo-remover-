import logging
import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

class MaskRefiner:
    def __init__(self, config):
        self.config = config

    def refine_masks(self, masks: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply morphological refinement, noise removal, and edge smoothing to all masks.
        """
        logger.info("Applying morphological refinement and noise removal to masks...")
        refined = []
        
        for idx, mask in enumerate(masks):
            # 1. Clean small components (noise)
            cleaned = self.clean_mask(mask, min_area=100)
            
            # 2. Morphological closing to fill holes inside the mask
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
            
            # 3. Edge smoothing (Gaussian blur then re-threshold)
            smoothed = cv2.GaussianBlur(closed, (5, 5), 0)
            _, thresh = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
            
            refined.append(thresh)
            
        logger.info("Mask refinement complete.")
        return refined

    def clean_mask(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """
        Remove small connected components below `min_area` from a binary mask.
        """
        # Connected component labeling
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels): # Start from 1 to skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255
                
        return cleaned

    def classify_motion(self, masks: list[np.ndarray], flows: list[np.ndarray] = None) -> str:
        """
        Determine if the segmented logo is 'static' or 'moving'.
        Checks intersection-over-union (IoU) of masks across frames.
        If masks remain mostly static (mean IoU > 0.95), we classify as 'static' (routes to LaMa).
        Otherwise, if the watermark moves or background slides behind it, we classify as 'moving' (routes to ProPainter).
        """
        logger.info("Classifying logo motion characteristics...")
        if len(masks) < 2:
            return 'static'
            
        ious = []
        for i in range(len(masks) - 1):
            m1 = masks[i]
            m2 = masks[i+1]
            
            intersection = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            
            if union == 0:
                iou = 1.0
            else:
                iou = intersection / float(union)
            ious.append(iou)
            
        mean_iou = np.mean(ious)
        logger.info(f"Mean temporal mask IoU: {mean_iou:.4f}")
        
        if mean_iou >= 0.94: # Very high overlap means static
            motion_type = 'static'
        else:
            motion_type = 'moving'
            
        logger.info(f"Logo classified as: {motion_type}")
        return motion_type

    def create_dilated_masks(self, masks: list[np.ndarray], dilation_px: int = None) -> list[np.ndarray]:
        """
        Apply dilation to masks with padding to guarantee full coverage of the logo and its boundary glow.
        """
        if dilation_px is None:
            dilation_px = self.config.MASK_DILATION_PX
            
        logger.info(f"Dilating masks by {dilation_px}px for final inpainting boundary padding...")
        dilated = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
        
        for mask in masks:
            dil = cv2.dilate(mask, kernel)
            dilated.append(dil)
            
        return dilated

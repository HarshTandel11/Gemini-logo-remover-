import os
import gc
import logging
import cv2
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

class SAM2Segmenter:
    def __init__(self, config, gpu_manager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.predictor = None

    def load(self):
        """
        Load SAM2 model onto GPU using GPUManager.
        """
        def load_fn():
            from sam2.build_sam import build_sam2_video_predictor
            # Check if checkpoint exists
            checkpoint_path = Path(self.config.SAM2_CHECKPOINT)
            if not checkpoint_path.exists():
                logger.info(f"SAM2 checkpoint not found at {checkpoint_path}. Downloading...")
                self._download_checkpoint()
            
            # Initialize predictor
            # Note: config file needs to be on the path or loaded correctly.
            # SAM2 expects model config file path or name.
            predictor = build_sam2_video_predictor(self.config.SAM2_MODEL_CFG, self.config.SAM2_CHECKPOINT)
            return predictor

        self.predictor = self.gpu_manager.load_model('sam2', load_fn)

    def _download_checkpoint(self):
        """
        Download SAM2 checkpoint if missing.
        """
        import urllib.request
        checkpoint_dir = Path(self.config.SAM2_CHECKPOINT).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_small.pt"
        logger.info(f"Downloading SAM2.1 Hiera Small checkpoint from {url}...")
        urllib.request.urlretrieve(url, self.config.SAM2_CHECKPOINT)
        logger.info("SAM2.1 checkpoint downloaded successfully.")

    def detect_logo_region(self, first_frame: np.ndarray) -> tuple[int, int, int, int]:
        """
        Auto-detect the logo region in the first frame.
        We check the corner regions (bottom-left and bottom-right quadrants)
        where watermarks are usually located.
        We search for high contrast, static transparent overlays, or edge density matching the Gemini logo.
        Returns:
            (x, y, w, h) bounding box.
        """
        logger.info("Auto-detecting logo region on first frame...")
        h, w = first_frame.shape[:2]
        
        # Define search areas: Bottom-left and Bottom-right corners
        # Watermarks typically occupy 10-30% of the width/height in corners
        quadrants = [
            # Bottom-right corner
            {"x_start": int(w * 0.7), "x_end": int(w * 0.98), "y_start": int(h * 0.7), "y_end": int(h * 0.98)},
            # Bottom-left corner
            {"x_start": int(w * 0.02), "x_end": int(w * 0.3), "y_start": int(h * 0.7), "y_end": int(h * 0.98)}
        ]
        
        best_bbox = (int(w * 0.75), int(h * 0.8), 120, 80) # Safe default bottom-right
        max_edge_density = 0
        
        # Convert frame to grayscale for edge analysis
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        # Apply bilateral filter to preserve logo edges but smooth out background noise
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        for quad in quadrants:
            quad_edges = edges[quad["y_start"]:quad["y_end"], quad["x_start"]:quad["x_end"]]
            # Find contours in the quadrant
            contours, _ = cv2.findContours(quad_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Ignore very small noise or too large boxes
                if 20 < cw < 300 and 20 < ch < 200:
                    aspect_ratio = cw / float(ch)
                    # The Gemini logo + text is usually wider than tall
                    if 0.5 < aspect_ratio < 3.0:
                        # Calculate edge density inside this contour box
                        cnt_edges = quad_edges[y:y+ch, x:x+cw]
                        density = np.sum(cnt_edges > 0) / float(cw * ch)
                        
                        if density > max_edge_density:
                            max_edge_density = density
                            # Convert back to global coordinates
                            best_bbox = (quad["x_start"] + x, quad["y_start"] + y, cw, ch)
        
        # Expand the bbox slightly to make sure we capture any glow/shadows around the logo
        x, y, cw, ch = best_bbox
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        cw = min(w - x, cw + 2 * padding)
        ch = min(h - y, ch + 2 * padding)
        
        logger.info(f"Detected logo bounding box: x={x}, y={y}, w={cw}, h={ch}")
        return (x, y, cw, ch)

    def segment_frames(self, frames_dir: str, logo_bbox: tuple) -> list[np.ndarray]:
        """
        Segment the logo across all frames using SAM2 video predictor.
        """
        if self.predictor is None:
            raise RuntimeError("SAM2 predictor not loaded. Call load() first.")

        logger.info(f"Segmenting video frames in {frames_dir} using SAM2...")
        
        # 1. Initialize inference state
        inference_state = self.predictor.init_state(video_path=frames_dir)
        
        # 2. Add bounding box prompt on frame 0
        obj_id = 1
        x, y, w, h = logo_bbox
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        
        logger.info(f"Adding bbox prompt on frame 0: {box}")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=box
        )
        
        # 3. Propagate in video
        # We collect all binary masks frame-by-frame
        masks = {}
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
            # mask_logits is of shape [1, H, W]
            mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            masks[frame_idx] = mask
            
        # Cleanup state
        self.predictor.reset_state(inference_state)
        
        # Convert dictionary to ordered list
        ordered_masks = [masks[i] for i in sorted(masks.keys())]
        logger.info(f"Generated segmentation masks for {len(ordered_masks)} frames.")
        return ordered_masks

    def unload(self):
        """
        Unload model and free GPU memory.
        """
        self.predictor = None
        self.gpu_manager.unload_model('sam2')

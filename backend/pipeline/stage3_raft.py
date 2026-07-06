import logging
import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

logger = logging.getLogger(__name__)

class RAFTTracker:
    def __init__(self, config, gpu_manager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.model = None

    def load(self):
        """
        Load RAFT model from torchvision.
        """
        def load_fn():
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=True).to(self.config.DEVICE)
            model.eval()
            return model

        self.model = self.gpu_manager.load_model('raft', load_fn)

    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between frame1 and frame2.
        Returns a flow field of shape (H, W, 2).
        """
        if self.model is None:
            raise RuntimeError("RAFT model not loaded. Call load() first.")

        # Convert to tensor and scale to [0, 1]
        t1 = torch.from_numpy(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        t2 = torch.from_numpy(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        # Normalise to [-1, 1] (RAFT expectation)
        t1 = (t1 - 0.5) * 2.0
        t2 = (t2 - 0.5) * 2.0

        # Add batch dimension and move to GPU
        t1 = t1.unsqueeze(0).to(self.config.DEVICE)
        t2 = t2.unsqueeze(0).to(self.config.DEVICE)

        # Pad height/width to be divisible by 8 (requirement for RAFT feature extraction)
        h, w = t1.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h), mode='replicate')
            t2 = torch.nn.functional.pad(t2, (0, pad_w, 0, pad_h), mode='replicate')

        with torch.no_grad():
            # Get flow predictions (returns list of refinement steps, last is best)
            flow_predictions = self.model(t1, t2)
            flow = flow_predictions[-1]

            # Crop back to original dimensions if padded
            if pad_h > 0 or pad_w > 0:
                flow = flow[..., :h, :w]

            # Move to CPU and format as numpy
            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
            return flow_np

    def compute_all_flows(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute forward optical flow between all consecutive frames.
        Returns a list of N-1 flow fields.
        """
        logger.info(f"Computing optical flow across {len(frames)} frames...")
        flows = []
        for i in range(len(frames) - 1):
            flow = self.compute_flow(frames[i], frames[i+1])
            flows.append(flow)
        logger.info("Optical flow computation complete.")
        return flows

    def warp_mask(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp a mask using the computed optical flow field.
        """
        h, w = mask.shape[:2]
        
        # Create pixel coordinate maps
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        
        # Warp mask using remap
        warped = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped

    def propagate_masks(self, masks: list[np.ndarray], flows: list[np.ndarray]) -> list[np.ndarray]:
        """
        Refine the masks using temporal optical flow tracking.
        Propagates masks forward using optical flow to fill in segmentation gaps or temporal stutter.
        """
        logger.info("Propagating and refining masks temporally...")
        refined_masks = [masks[0].copy()]
        
        for i in range(1, len(masks)):
            flow = flows[i-1]
            prev_refined = refined_masks[-1]
            current_mask = masks[i]
            
            # Warp the previous refined mask forward
            warped_prev = self.warp_mask(prev_refined, flow)
            
            # Combine current SAM2 segment with flow-warped previous mask
            # Using logical OR ensures tracking continuity in case SAM2 drops detection
            combined = cv2.bitwise_or(current_mask, warped_prev)
            
            # Soft threshold or keep as binary mask
            _, binary_combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
            refined_masks.append(binary_combined)
            
        logger.info("Temporal mask propagation complete.")
        return refined_masks

    def unload(self):
        """
        Unload model and free GPU memory.
        """
        self.model = None
        self.gpu_manager.unload_model('raft')

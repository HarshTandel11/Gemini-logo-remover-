"""
Stage 5b: ProPainter Video Inpainting

Uses the ProPainter model for video-aware inpainting that leverages
temporal information across frames for superior coherence in dynamic
scenes. Invoked via subprocess to call ProPainter's inference script.
"""

import glob
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default ProPainter directory - can be overridden via environment variable
PROPAINTER_DIR = os.environ.get(
    "PROPAINTER_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ProPainter"),
)


class ProPainterInpainter:
    """ProPainter-based video inpainting for temporally coherent logo removal.

    Uses ProPainter's flow-guided video completion approach, which considers
    temporal context across multiple frames for high-quality results on
    dynamic backgrounds and moving cameras.
    """

    def __init__(self, propainter_dir: Optional[str] = None) -> None:
        """Initialize ProPainterInpainter.

        Args:
            propainter_dir: Path to the ProPainter repository root.
                            Defaults to PROPAINTER_DIR environment variable
                            or a sibling directory.
        """
        self._propainter_dir = Path(propainter_dir or PROPAINTER_DIR).resolve()
        self._weights_dir = self._propainter_dir / "weights"
        self._inference_script = self._propainter_dir / "inference_propainter.py"
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if ProPainter is ready to use."""
        return self._loaded

    def load(self) -> None:
        """Verify that ProPainter weights and inference script exist.

        Raises:
            FileNotFoundError: If the ProPainter directory, weights, or
                               inference script are not found.
        """
        if self._loaded:
            logger.debug("ProPainter already verified, skipping.")
            return

        logger.info("Verifying ProPainter installation at: %s", self._propainter_dir)

        # Check ProPainter directory
        if not self._propainter_dir.is_dir():
            raise FileNotFoundError(
                f"ProPainter directory not found: {self._propainter_dir}. "
                f"Set PROPAINTER_DIR environment variable or provide the path."
            )

        # Check inference script
        if not self._inference_script.is_file():
            raise FileNotFoundError(
                f"ProPainter inference script not found: {self._inference_script}"
            )

        # Check weights directory
        if not self._weights_dir.is_dir():
            raise FileNotFoundError(
                f"ProPainter weights directory not found: {self._weights_dir}. "
                f"Download weights and place them in {self._weights_dir}/"
            )

        # Verify at least some weight files exist
        weight_files = list(self._weights_dir.glob("*.pth")) + list(
            self._weights_dir.glob("*.pt")
        )
        if not weight_files:
            raise FileNotFoundError(
                f"No model weight files (.pth/.pt) found in {self._weights_dir}. "
                f"Please download ProPainter weights."
            )

        logger.info(
            "ProPainter verified: %d weight files found.", len(weight_files)
        )
        self._loaded = True

    def inpaint_video(
        self,
        frames_dir: str,
        masks_dir: str,
        output_dir: str,
        height: int = 480,
        width: int = 854,
        fp16: bool = True,
        neighbor_length: int = 10,
        subvideo_length: int = 80,
    ) -> List[np.ndarray]:
        """Run ProPainter video inpainting via subprocess.

        Calls ProPainter's inference_propainter.py script with the specified
        parameters, then reads the output frames back as numpy arrays.

        Args:
            frames_dir: Directory containing input frames as PNG/JPG images.
            masks_dir: Directory containing mask images corresponding to frames.
            output_dir: Directory where ProPainter will write output frames.
            height: Resize height for processing (default 480).
            width: Resize width for processing (default 854).
            fp16: Use half-precision for faster inference (default True).
            neighbor_length: Number of local neighbor frames for flow completion
                             (default 10).
            subvideo_length: Length of sub-videos for processing long videos
                             (default 80).

        Returns:
            List of inpainted frames as BGR numpy arrays (H, W, 3), uint8.

        Raises:
            RuntimeError: If ProPainter is not loaded or subprocess fails.
            FileNotFoundError: If no output frames are produced.
        """
        if not self._loaded:
            raise RuntimeError("ProPainter not loaded. Call load() first.")

        # Validate input directories
        frames_path = Path(frames_dir)
        masks_path = Path(masks_dir)
        output_path = Path(output_dir)

        if not frames_path.is_dir():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        if not masks_path.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Build the ProPainter command
        cmd = [
            sys.executable,
            str(self._inference_script),
            "--video", str(frames_path),
            "--mask", str(masks_path),
            "--output", str(output_path),
            "--height", str(height),
            "--width", str(width),
            "--neighbor_length", str(neighbor_length),
            "--subvideo_length", str(subvideo_length),
        ]

        if fp16:
            cmd.append("--fp16")

        logger.info("Running ProPainter: %s", " ".join(cmd))

        # Execute ProPainter subprocess
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self._propainter_dir),
                timeout=3600,  # 1 hour timeout for long videos
                check=False,
            )

            if result.stdout:
                logger.debug("ProPainter stdout:\n%s", result.stdout[-2000:])
            if result.stderr:
                logger.debug("ProPainter stderr:\n%s", result.stderr[-2000:])

            if result.returncode != 0:
                raise RuntimeError(
                    f"ProPainter exited with code {result.returncode}. "
                    f"stderr: {result.stderr[-500:]}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ProPainter timed out after 3600 seconds.")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Failed to execute ProPainter: {e}. "
                f"Ensure Python is available and ProPainter is installed."
            ) from e

        # Read output frames
        output_frames = self._read_output_frames(output_path)

        if not output_frames:
            raise FileNotFoundError(
                f"No output frames found in {output_path}. "
                f"ProPainter may have failed silently."
            )

        logger.info(
            "ProPainter inpainting complete: %d frames produced.", len(output_frames)
        )
        return output_frames

    @staticmethod
    def _read_output_frames(output_dir: Path) -> List[np.ndarray]:
        """Read output frames from a directory, sorted by filename.

        Searches for common ProPainter output patterns including subdirectories
        like 'results/' or 'inpaint_result/'.

        Args:
            output_dir: Directory to search for output frame images.

        Returns:
            Sorted list of frames as BGR numpy arrays.
        """
        # ProPainter may place results in a subdirectory
        search_dirs = [output_dir]
        for subdir_name in ["results", "inpaint_result", "result"]:
            subdir = output_dir / subdir_name
            if subdir.is_dir():
                search_dirs.insert(0, subdir)

        frame_paths: List[str] = []
        for search_dir in search_dirs:
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                frame_paths.extend(glob.glob(str(search_dir / ext)))
            if frame_paths:
                break

        # Sort by filename to ensure correct frame order
        frame_paths.sort(key=lambda p: os.path.basename(p))

        frames: List[np.ndarray] = []
        for path in frame_paths:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
            else:
                logger.warning("Failed to read output frame: %s", path)

        return frames

    def unload(self) -> None:
        """Mark ProPainter as unloaded.

        ProPainter runs as a subprocess, so there's no in-process model to
        release. This method resets the loaded state.
        """
        if self._loaded:
            logger.info("Unloading ProPainter (resetting state).")
            self._loaded = False
        else:
            logger.debug("ProPainter not loaded, nothing to unload.")

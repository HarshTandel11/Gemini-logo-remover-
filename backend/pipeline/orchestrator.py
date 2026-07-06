"""
Stage Orchestrator — Central Pipeline Coordinator

Runs all ten stages of the video-logo-removal pipeline sequentially,
managing GPU memory between stages and reporting progress via callbacks.

Stages:
  1. Scene detection & frame extraction
  2. SAM2 segmentation (logo mask generation)
  3. RAFT optical-flow tracking & mask propagation
  4. Mask refinement, motion classification, dilation
  5a. LaMa inpainting (static masks)
  5b. ProPainter inpainting (dynamic / moving masks)
  6. DiffuEraser confidence-aware refinement
  7. SDXL inpainting (low-confidence fallback)
  8. FRESCO temporal enhancement
  9. Laplacian blending
 10. Video encoding
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

import backend.config as config
from backend.config import (
    DEFAULT_OUTPUT_CODEC,
    DEFAULT_OUTPUT_CRF,
    MASK_DILATION_PX,
    OUTPUT_DIR,
    SDXL_CONFIDENCE_THRESHOLD,
    SDXL_ENABLED,
    TEMP_DIR,
)
from backend.utils.gpu_manager import GPUManager

logger = logging.getLogger(__name__)

# Type alias for the progress callback.
ProgressCallback = Callable[[str, float, str], None]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract the audio track from *video_path* using ffmpeg.

    Returns ``True`` if an audio stream was found and extracted, else ``False``.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "copy",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and audio_path.exists():
            logger.info("Audio extracted → %s", audio_path)
            return True
        logger.info("No audio stream found or extraction failed.")
        return False
    except FileNotFoundError:
        logger.warning("ffmpeg not found on PATH; skipping audio extraction.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Audio extraction timed out.")
        return False


def _get_video_info(video_path: Path) -> Dict[str, Any]:
    """Return basic metadata for *video_path* via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def _decode_video(video_path: Path) -> List[np.ndarray]:
    """Read every frame of *video_path* into a list of BGR arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    logger.info("Decoded %d frames from %s", len(frames), video_path.name)
    return frames


def _save_frames_to_dir(frames: List[np.ndarray], directory: Path) -> None:
    """Write *frames* as sequentially-numbered JPEGs inside *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(directory / f"{i:06d}.jpg"), frame)


# ── Orchestrator ─────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    """Central coordinator that runs all pipeline stages in order."""

    STAGE_NAMES: List[str] = [
        "scene_detection",
        "segmentation",
        "tracking",
        "mask_refinement",
        "inpainting",
        "diffueraser",
        "sdxl_refinement",
        "temporal_enhancement",
        "blending",
        "encoding",
    ]

    def __init__(self) -> None:
        self._gpu = GPUManager()
        self._progress_cb: Optional[ProgressCallback] = None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logger.info("PipelineOrchestrator initialised.")

    # ── Progress reporting ────────────────────────────────────────────────

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Register a ``callback(stage, progress, message)`` function."""
        self._progress_cb = callback

    def _report(self, stage: str, progress: float, message: str) -> None:
        """Emit a progress update (0.0 – 1.0) to the registered callback."""
        logger.info("[%s] %.0f%% – %s", stage, progress * 100, message)
        if self._progress_cb is not None:
            try:
                self._progress_cb(stage, progress, message)
            except Exception:
                logger.debug("Progress callback raised; ignoring.", exc_info=True)

    # ── Main entry point ──────────────────────────────────────────────────

    def process(
        self,
        video_path: str,
        output_path: str | None = None,
        logo_bbox: Tuple[int, int, int, int] | None = None,
    ) -> str:
        """Run the full pipeline on *video_path* and return the output path.

        Args:
            video_path: Path to the input video file.
            output_path: Optional explicit output path.  When ``None`` a
                timestamped filename is generated under ``OUTPUT_DIR``.
            logo_bbox: Optional manual bounding box for the logo (x, y, w, h).

        Returns:
            Absolute path to the processed video file.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: On any unrecoverable pipeline error.
        """
        src = Path(video_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Input video not found: {src}")

        if output_path is None:
            ts = int(time.time())
            output_path = str(OUTPUT_DIR / f"{src.stem}_clean_{ts}.mp4")
        dst = Path(output_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Temporary working directory (auto-cleaned on success)
        work_dir = Path(tempfile.mkdtemp(prefix="logorm_", dir=str(TEMP_DIR)))
        frames_dir = work_dir / "frames"
        frames_dir.mkdir()

        logger.info("Pipeline started: %s → %s", src, dst)
        logger.info("Working directory: %s", work_dir)

        try:
            # ── Pre-processing ────────────────────────────────────────────
            video_info = _get_video_info(src)
            fps: float = video_info["fps"]
            logger.info(
                "Video: %dx%d @ %.2f fps, %d frames",
                video_info["width"], video_info["height"],
                fps, video_info["frame_count"],
            )

            audio_path = work_dir / "audio.aac"
            has_audio = _extract_audio(src, audio_path)

            # ── Stage 1: Scene detection & frame decoding ─────────────────
            self._report("scene_detection", 0.0, "Decoding video frames…")
            frames = _decode_video(src)
            total_frames = len(frames)

            self._report("scene_detection", 0.3, "Detecting scene boundaries…")
            scene_list = self._detect_scenes(src)
            self._report("scene_detection", 1.0, f"Found {len(scene_list)} scene(s).")

            # Save frames to disk for SAM2 (it reads from a directory)
            self._report("segmentation", 0.0, "Saving frames for SAM2…")
            _save_frames_to_dir(frames, frames_dir)

            # ── Stage 2: SAM2 Segmentation ────────────────────────────────
            self._report("segmentation", 0.1, "Loading SAM2 model…")
            logo_region, masks = self._run_segmentation(frames_dir, frames, logo_bbox=logo_bbox)
            self._report("segmentation", 1.0, "Logo segmentation complete.")

            # ── Stage 3: RAFT Optical-flow Tracking ───────────────────────
            self._report("tracking", 0.0, "Loading RAFT model…")
            flows, masks = self._run_tracking(frames, masks)
            self._report("tracking", 1.0, "Tracking & mask propagation complete.")

            # ── Stage 4: Mask Refinement ──────────────────────────────────
            self._report("mask_refinement", 0.0, "Refining masks…")
            masks, motion_types, dilated_masks = self._run_mask_refinement(
                masks, flows
            )
            self._report("mask_refinement", 1.0, "Mask refinement complete.")

            # ── Stage 5a/5b: Inpainting (LaMa or ProPainter) ─────────────
            self._report("inpainting", 0.0, "Starting inpainting…")
            inpainted = self._run_inpainting(
                frames, dilated_masks, motion_types, scene_list
            )
            self._report("inpainting", 1.0, "Inpainting complete.")

            # ── Stage 6: DiffuEraser Refinement ───────────────────────────
            self._report("diffueraser", 0.0, "Running DiffuEraser refinement…")
            refined, confidence = self._run_diffueraser(
                inpainted, frames, dilated_masks
            )
            self._report(
                "diffueraser",
                1.0,
                f"DiffuEraser done (avg confidence {confidence:.2f}).",
            )

            # ── Stage 7: SDXL Fallback (low confidence only) ─────────────
            if SDXL_ENABLED and confidence < SDXL_CONFIDENCE_THRESHOLD:
                self._report("sdxl_refinement", 0.0, "SDXL refinement triggered…")
                refined = self._run_sdxl(refined, dilated_masks)
                self._report("sdxl_refinement", 1.0, "SDXL refinement complete.")
            else:
                reason = (
                    "disabled" if not SDXL_ENABLED
                    else f"confidence {confidence:.2f} ≥ {SDXL_CONFIDENCE_THRESHOLD}"
                )
                self._report(
                    "sdxl_refinement", 1.0, f"Skipped SDXL ({reason})."
                )

            # ── Stage 8: FRESCO Temporal Enhancement ──────────────────────
            self._report("temporal_enhancement", 0.0, "Enhancing temporal coherence…")
            enhanced = self._run_fresco(refined, flows)
            self._report("temporal_enhancement", 1.0, "Temporal enhancement complete.")

            # ── Stage 9: Laplacian Blending ───────────────────────────────
            self._report("blending", 0.0, "Blending results…")
            blended = self._run_blending(enhanced, frames, dilated_masks)
            self._report("blending", 1.0, "Blending complete.")

            # ── Stage 10: Video Encoding ──────────────────────────────────
            self._report("encoding", 0.0, "Encoding output video…")
            self._encode_video(
                blended,
                fps=fps,
                audio_path=audio_path if has_audio else None,
                output_path=dst,
            )
            self._report("encoding", 1.0, "Encoding complete.")

            logger.info("Pipeline finished: %s", dst)
            return str(dst)

        except Exception:
            logger.exception("Pipeline failed!")
            raise
        finally:
            # Always clean up GPU resources
            self._gpu.unload_all()
            # Clean up temp directory on success (keep on failure for debugging)
            if dst.is_file():
                shutil.rmtree(work_dir, ignore_errors=True)
                logger.info("Cleaned up temp directory: %s", work_dir)

    # ── Stage implementations (private) ───────────────────────────────────

    def _detect_scenes(self, video_path: Path) -> List[Tuple[int, int]]:
        """Stage 1: detect scene boundaries using PySceneDetect."""
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector

            from backend.config import SCENE_THRESHOLD

            video = open_video(str(video_path))
            scene_mgr = SceneManager()
            scene_mgr.add_detector(ContentDetector(threshold=SCENE_THRESHOLD))
            scene_mgr.detect_scenes(video)
            raw_scenes = scene_mgr.get_scene_list()

            scenes: List[Tuple[int, int]] = []
            for start_ts, end_ts in raw_scenes:
                scenes.append((start_ts.get_frames(), end_ts.get_frames()))

            if not scenes:
                # Treat entire video as one scene
                cap = cv2.VideoCapture(str(video_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                scenes = [(0, total)]

            logger.info("Detected %d scene(s).", len(scenes))
            return scenes

        except ImportError:
            logger.warning(
                "scenedetect not installed; treating video as a single scene."
            )
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return [(0, total)]

    def _run_segmentation(
        self,
        frames_dir: Path,
        frames: List[np.ndarray],
        logo_bbox: Tuple[int, int, int, int] | None = None,
    ) -> Tuple[Dict[str, int], List[np.ndarray]]:
        """Stage 2: SAM2 logo region detection and frame segmentation."""
        try:
            from backend.pipeline.stage2_sam2 import SAM2Segmenter

            segmenter = SAM2Segmenter(config, self._gpu)
            with self._gpu.stage("sam2"):
                segmenter.load()
                try:
                    if logo_bbox is None:
                        bbox_tuple = segmenter.detect_logo_region(frames[0])
                    else:
                        bbox_tuple = logo_bbox
                    
                    logo_region = {
                        "x": bbox_tuple[0],
                        "y": bbox_tuple[1],
                        "w": bbox_tuple[2],
                        "h": bbox_tuple[3]
                    }
                    self._report(
                        "segmentation", 0.4,
                        f"Logo region: {logo_region}",
                    )
                    masks = segmenter.segment_frames(
                        str(frames_dir), bbox_tuple
                    )
                    self._report("segmentation", 0.9, "Segmentation complete.")
                finally:
                    segmenter.unload()
                    self._gpu.unload_model("sam2") if "sam2" in self._gpu._models else None

            return logo_region, masks

        except ImportError:
            logger.warning(
                "SAM2 segmenter not available; generating placeholder masks."
            )
            h, w = frames[0].shape[:2]
            placeholder = np.zeros((h, w), dtype=np.uint8)
            return {"x": 0, "y": 0, "w": w, "h": h}, [placeholder] * len(frames)

    def _run_tracking(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Stage 3: RAFT optical-flow computation and mask propagation."""
        try:
            from backend.pipeline.stage3_raft import RAFTTracker

            tracker = RAFTTracker(config, self._gpu)
            with self._gpu.stage("raft"):
                tracker.load()
                try:
                    flows = tracker.compute_all_flows(frames)
                    self._report("tracking", 0.6, "Optical flow computed.")
                    masks = tracker.propagate_masks(masks, flows)
                    self._report("tracking", 0.9, "Masks propagated.")
                finally:
                    tracker.unload()

            return flows, masks

        except ImportError:
            logger.warning("RAFT tracker not available; skipping flow computation.")
            # Return empty flows and keep masks unchanged
            h, w = frames[0].shape[:2]
            empty_flow = np.zeros((h, w, 2), dtype=np.float32)
            return [empty_flow] * max(len(frames) - 1, 1), masks

    def _run_mask_refinement(
        self,
        masks: List[np.ndarray],
        flows: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[str], List[np.ndarray]]:
        """Stage 4: refine masks, classify motion, create dilated versions."""
        try:
            from backend.pipeline.stage4_mask_refiner import MaskRefiner

            refiner = MaskRefiner(config)
            refined_masks = refiner.refine_masks(masks)
            self._report("mask_refinement", 0.4, "Masks refined.")

            motion_types = refiner.classify_motion(refined_masks, flows)
            self._report("mask_refinement", 0.7, f"Motion classified: {set(motion_types)}")

            dilated_masks = refiner.create_dilated_masks(
                refined_masks, dilation_px=MASK_DILATION_PX
            )
            return refined_masks, motion_types, dilated_masks

        except ImportError:
            logger.warning(
                "MaskRefiner not available; using raw masks with dilation."
            )
            dilated: List[np.ndarray] = []
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (MASK_DILATION_PX * 2 + 1, MASK_DILATION_PX * 2 + 1),
            )
            for m in masks:
                dilated.append(cv2.dilate(m, kernel, iterations=1))
            motion_types = ["static"] * len(masks)
            return masks, motion_types, dilated

    def _run_inpainting(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        motion_types: List[str],
        scene_list: List[Tuple[int, int]],
    ) -> List[np.ndarray]:
        """Stage 5a/5b: select LaMa or ProPainter per scene based on motion."""
        inpainted = list(frames)  # shallow copy

        for scene_idx, (start, end) in enumerate(scene_list):
            scene_frames = frames[start:end]
            scene_masks = masks[start:end]
            scene_motions = motion_types[start:end]

            # Determine dominant motion type for this scene
            dynamic_count = sum(1 for m in scene_motions if m == "dynamic")
            is_dynamic = dynamic_count > len(scene_motions) * 0.3

            if is_dynamic:
                scene_result = self._inpaint_propainter(
                    scene_frames, scene_masks, scene_idx
                )
            else:
                scene_result = self._inpaint_lama(
                    scene_frames, scene_masks, scene_idx
                )

            inpainted[start:end] = scene_result

            progress = (scene_idx + 1) / len(scene_list)
            self._report(
                "inpainting",
                progress * 0.9,
                f"Scene {scene_idx + 1}/{len(scene_list)} inpainted.",
            )

        return inpainted

    def _inpaint_lama(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        scene_idx: int,
    ) -> List[np.ndarray]:
        """Run LaMa inpainting on a batch of frames."""
        try:
            from backend.pipeline.stage5a_lama import LaMaInpainter

            inpainter = LaMaInpainter()
            with self._gpu.stage(f"lama_scene{scene_idx}"):
                inpainter.load()
                try:
                    result = inpainter.inpaint_batch(frames, masks)
                finally:
                    inpainter.unload()
            return result

        except ImportError:
            logger.warning(
                "LaMa inpainter not available; returning original frames."
            )
            return list(frames)

    def _inpaint_propainter(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        scene_idx: int,
    ) -> List[np.ndarray]:
        """Run ProPainter video inpainting on a batch of frames."""
        try:
            from backend.pipeline.stage5b_propainter import ProPainterInpainter

            inpainter = ProPainterInpainter(propainter_dir=str(config.PROPAINTER_DIR))
            with self._gpu.stage(f"propainter_scene{scene_idx}"):
                inpainter.load()
                try:
                    result = inpainter.inpaint_batch(frames, masks)
                finally:
                    inpainter.unload()
            return result

        except ImportError:
            logger.warning(
                "ProPainter not available; falling back to LaMa."
            )
            return self._inpaint_lama(frames, masks, scene_idx)

    def _run_diffueraser(
        self,
        inpainted: List[np.ndarray],
        originals: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], float]:
        """Stage 6: DiffuEraser refinement with confidence scoring."""
        if not getattr(config, "DIFFUERASER_ENABLED", False):
            logger.info("DiffuEraser refinement is disabled in config. Skipping.")
            return inpainted, 1.0

        try:
            from backend.pipeline.stage6_diffueraser import DiffuEraserRefiner

            refiner = DiffuEraserRefiner()
            with self._gpu.stage("diffueraser"):
                refiner.load()
                try:
                    refined = refiner.refine_batch(inpainted, originals, masks)
                    self._report("diffueraser", 0.8, "Refinement complete.")
                    confidence = refiner.compute_confidence(
                        refined, originals, masks
                    )
                finally:
                    refiner.unload()
            return refined, confidence

        except ImportError:
            logger.warning(
                "DiffuEraser not available; skipping refinement."
            )
            return inpainted, 1.0  # High confidence to skip SDXL

    def _run_sdxl(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Stage 7: SDXL inpainting for low-confidence regions."""
        try:
            from backend.pipeline.stage7_sdxl import SDXLInpainter

            inpainter = SDXLInpainter()
            with self._gpu.stage("sdxl"):
                inpainter.load()
                try:
                    result = inpainter.inpaint_batch(frames, masks)
                finally:
                    inpainter.unload()
            return result

        except ImportError:
            logger.warning(
                "SDXL inpainter not available; skipping."
            )
            return frames

    def _run_fresco(
        self,
        frames: List[np.ndarray],
        flows: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Stage 8: FRESCO temporal consistency enhancement."""
        try:
            from backend.pipeline.stage8_fresco import FRESCOEnhancer

            enhancer = FRESCOEnhancer()
            with self._gpu.stage("fresco"):
                enhancer.load()
                try:
                    result = enhancer.enhance(frames, flows)
                finally:
                    enhancer.unload()
            return result

        except ImportError:
            logger.warning(
                "FRESCO enhancer not available; skipping temporal enhancement."
            )
            return frames

    def _run_blending(
        self,
        processed: List[np.ndarray],
        originals: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Stage 9: Laplacian pyramid blending."""
        try:
            from backend.pipeline.stage9_blender import LaplacianBlender

            blender = LaplacianBlender()
            result = blender.blend_batch(processed, originals, masks)
            return result

        except ImportError:
            logger.warning(
                "LaplacianBlender not available; using direct mask compositing."
            )
            blended: List[np.ndarray] = []
            for proc, orig, mask in zip(processed, originals, masks):
                m = mask.astype(np.float32) / 255.0
                if m.ndim == 2:
                    m = m[:, :, np.newaxis]
                composite = (proc * m + orig * (1.0 - m)).astype(np.uint8)
                blended.append(composite)
            return blended

    def _encode_video(
        self,
        frames: List[np.ndarray],
        fps: float,
        audio_path: Optional[Path],
        output_path: Path,
    ) -> None:
        """Stage 10: encode processed frames (and optional audio) to video."""
        try:
            from backend.pipeline.stage10_encoder import VideoEncoder

            encoder = VideoEncoder()
            encoder.encode(
                frames,
                output_path=str(output_path),
                fps=fps,
                audio_source=str(audio_path) if audio_path else None,
            )

        except ImportError:
            logger.warning(
                "VideoEncoder not available; using OpenCV VideoWriter fallback."
            )
            self._encode_opencv_fallback(frames, fps, audio_path, output_path)

    def _encode_opencv_fallback(
        self,
        frames: List[np.ndarray],
        fps: float,
        audio_path: Optional[Path],
        output_path: Path,
    ) -> None:
        """Fallback encoder using OpenCV + ffmpeg audio mux."""
        if not frames:
            raise RuntimeError("No frames to encode.")

        h, w = frames[0].shape[:2]
        temp_video = output_path.with_suffix(".tmp.mp4")

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (w, h))

        for i, frame in enumerate(frames):
            writer.write(frame)
            if (i + 1) % 100 == 0:
                self._report(
                    "encoding",
                    (i + 1) / len(frames) * 0.8,
                    f"Written {i + 1}/{len(frames)} frames.",
                )
        writer.release()

        # Mux audio back if available
        if audio_path and audio_path.is_file():
            self._report("encoding", 0.85, "Muxing audio…")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(temp_video),
                        "-i", str(audio_path),
                        "-c:v", DEFAULT_OUTPUT_CODEC,
                        "-crf", str(DEFAULT_OUTPUT_CRF),
                        "-c:a", "aac",
                        "-shortest",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=True,
                )
                temp_video.unlink(missing_ok=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning(
                    "ffmpeg mux failed; output will have no audio."
                )
                shutil.move(str(temp_video), str(output_path))
        else:
            # Re-encode with proper codec via ffmpeg, or just rename
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(temp_video),
                        "-c:v", DEFAULT_OUTPUT_CODEC,
                        "-crf", str(DEFAULT_OUTPUT_CRF),
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=True,
                )
                temp_video.unlink(missing_ok=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                shutil.move(str(temp_video), str(output_path))

        self._report("encoding", 0.95, "Encoding finalised.")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Logo Remover — Pipeline Orchestrator",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the output video (default: auto-generated).",
    )
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator()

    def cli_progress(stage: str, progress: float, message: str) -> None:
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {progress*100:5.1f}%  {stage}: {message}", end="", flush=True)
        if progress >= 1.0:
            print()

    orchestrator.set_progress_callback(cli_progress)

    result_path = orchestrator.process(args.input, args.output)
    print(f"\n✓ Output saved to: {result_path}")

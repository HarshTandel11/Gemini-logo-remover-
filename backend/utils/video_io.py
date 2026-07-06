"""
Video I/O utilities.

Provides helpers for frame extraction, frame saving, metadata retrieval,
audio extraction, and audio–video muxing.  Heavy lifting is done by
OpenCV for pixel-level work and FFmpeg (via subprocess) for container /
codec operations.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Locate the ``ffmpeg`` binary on the system PATH.

    Returns:
        Absolute path to ``ffmpeg``.

    Raises:
        FileNotFoundError: If ``ffmpeg`` is not installed or not on PATH.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/"
        )
    return path


def _find_ffprobe() -> str:
    """Locate the ``ffprobe`` binary on the system PATH.

    Returns:
        Absolute path to ``ffprobe``.

    Raises:
        FileNotFoundError: If ``ffprobe`` is not installed or not on PATH.
    """
    path = shutil.which("ffprobe")
    if path is None:
        raise FileNotFoundError(
            "ffprobe not found on PATH. Install it from https://ffmpeg.org/"
        )
    return path


# ── Frame extraction / saving ─────────────────────────────────────────────────


def extract_frames(video_path: Union[str, Path]) -> List[np.ndarray]:
    """Read every frame from *video_path* into a list of NumPy arrays.

    Frames are returned in **BGR** colour order (OpenCV convention).

    Args:
        video_path: Path to a video file readable by OpenCV.

    Returns:
        List of ``(H, W, 3)`` uint8 NumPy arrays, one per frame.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If OpenCV cannot open the video.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()

    logger.info("Extracted %d frames from %s", len(frames), video_path.name)
    return frames


def save_frames(
    frames: List[np.ndarray],
    output_dir: Union[str, Path],
    prefix: str = "frame",
    ext: str = ".png",
) -> List[Path]:
    """Save a list of frames as individual image files.

    Args:
        frames: List of ``(H, W, 3)`` BGR uint8 arrays.
        output_dir: Directory where images will be written.  Created if it
            does not exist.
        prefix: Filename prefix.  Files are named
            ``{prefix}_{index:06d}{ext}``.
        ext: Image extension (e.g. ``".png"``, ``".jpg"``).

    Returns:
        List of :class:`Path` objects pointing to the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for idx, frame in enumerate(frames):
        filepath = output_dir / f"{prefix}_{idx:06d}{ext}"
        cv2.imwrite(str(filepath), frame)
        paths.append(filepath)

    logger.info("Saved %d frames to %s", len(paths), output_dir)
    return paths


# ── Video metadata ────────────────────────────────────────────────────────────


def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Return metadata for *video_path*.

    Uses ``ffprobe`` (JSON output) for accurate container-level metadata.
    Falls back to OpenCV if ``ffprobe`` is unavailable.

    Returns:
        A dict with keys:

        - ``fps`` (float)
        - ``width`` (int)
        - ``height`` (int)
        - ``frame_count`` (int)
        - ``duration`` (float, seconds)
        - ``codec`` (str)
        - ``has_audio`` (bool)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        return _get_video_info_ffprobe(video_path)
    except FileNotFoundError:
        logger.warning("ffprobe unavailable – falling back to OpenCV for metadata.")
        return _get_video_info_cv2(video_path)


def _get_video_info_ffprobe(video_path: Path) -> Dict[str, Any]:
    """Retrieve video info via ``ffprobe``."""
    ffprobe = _find_ffprobe()
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=30
    )
    probe: Dict[str, Any] = json.loads(result.stdout)

    # Locate the video stream.
    video_stream: Optional[Dict[str, Any]] = None
    has_audio = False
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        if stream.get("codec_type") == "audio":
            has_audio = True

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    # Parse FPS from the r_frame_rate fraction (e.g. "30000/1001").
    r_frame_rate: str = video_stream.get("r_frame_rate", "30/1")
    num, den = (int(x) for x in r_frame_rate.split("/"))
    fps = num / den if den else 30.0

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    frame_count = int(video_stream.get("nb_frames", 0))

    # Duration: prefer stream-level, then container-level.
    duration = float(
        video_stream.get(
            "duration",
            probe.get("format", {}).get("duration", 0),
        )
    )

    # If nb_frames was missing, estimate from duration × fps.
    if frame_count == 0 and duration > 0:
        frame_count = int(round(duration * fps))

    codec = video_stream.get("codec_name", "unknown")

    info: Dict[str, Any] = {
        "fps": round(fps, 4),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": round(duration, 4),
        "codec": codec,
        "has_audio": has_audio,
    }
    logger.info("Video info (ffprobe): %s", info)
    return info


def _get_video_info_cv2(video_path: Path) -> Dict[str, Any]:
    """Fallback metadata extraction via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()

    info: Dict[str, Any] = {
        "fps": round(fps, 4),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": round(duration, 4),
        "codec": "unknown",
        "has_audio": False,  # OpenCV cannot detect audio streams.
    }
    logger.info("Video info (cv2): %s", info)
    return info


# ── Audio extraction / muxing ─────────────────────────────────────────────────


def extract_audio(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Extract the audio stream from *video_path* without re-encoding.

    Args:
        video_path: Source video file.
        output_path: Destination for the extracted audio (e.g. ``.aac``).

    Returns:
        :class:`Path` to the written audio file.

    Raises:
        FileNotFoundError: If *video_path* does not exist or ``ffmpeg`` is
            missing.
        subprocess.CalledProcessError: If FFmpeg exits with non-zero status
            (e.g. no audio stream present).
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",                # Overwrite without asking.
        "-i", str(video_path),
        "-vn",               # Drop video.
        "-acodec", "copy",   # Copy audio codec (no re-encode).
        str(output_path),
    ]
    logger.info("Extracting audio: %s -> %s", video_path.name, output_path.name)
    subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
    logger.info("Audio extracted: %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


def merge_audio_video(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    codec: str = "copy",
) -> Path:
    """Mux a video file with an audio file into a single container.

    Both streams are copied without re-encoding by default.

    Args:
        video_path: Video-only (or silent-video) source.
        audio_path: Audio source.
        output_path: Final muxed output file.
        codec: Video codec to use.  ``"copy"`` avoids re-encoding.

    Returns:
        :class:`Path` to the written output file.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", codec,
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    logger.info(
        "Merging audio + video: %s + %s -> %s",
        video_path.name,
        audio_path.name,
        output_path.name,
    )
    subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    logger.info("Merged output: %s (%.1f MB)", output_path.name, output_path.stat().st_size / 1024**2)
    return output_path


def frames_to_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: float = 30.0,
    codec: str = "mp4v",
) -> Path:
    """Encode a list of BGR frames into a video file via OpenCV.

    Args:
        frames: List of ``(H, W, 3)`` uint8 BGR arrays.
        output_path: Destination video file.
        fps: Output frame rate.
        codec: FourCC codec string (default ``mp4v``).

    Returns:
        :class:`Path` to the written video.
    """
    if not frames:
        raise ValueError("No frames provided to encode.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    logger.info(
        "Wrote %d frames to %s at %.2f fps (%dx%d)",
        len(frames), output_path.name, fps, w, h,
    )
    return output_path

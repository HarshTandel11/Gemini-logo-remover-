"""
Stage 10: Final Video Output

Encodes processed frames into a final video file using FFmpeg.
Supports configurable codec, CRF quality, and optional audio merging
from the original source video.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoEncoder:
    """FFmpeg-based video encoder for final output.

    Saves processed frames as temporary PNGs, encodes them into a video
    using FFmpeg with configurable codec and quality settings, and
    optionally merges audio from the original source video.
    """

    def encode(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float,
        audio_source: Optional[str] = None,
        codec: str = "libx264",
        crf: int = 18,
        pix_fmt: str = "yuv420p",
    ) -> str:
        """Encode frames into a video file.

        Saves frames as temporary PNGs, then uses FFmpeg to encode them
        into the output video. If an audio source is provided, the audio
        track is merged into the final output.

        Args:
            frames: List of BGR numpy arrays (H, W, 3), uint8.
            output_path: Path for the output video file.
            fps: Frame rate of the output video.
            audio_source: Optional path to a video/audio file from which
                          to copy the audio track.
            codec: FFmpeg video codec (default 'libx264').
            crf: Constant Rate Factor for quality (default 18, lower = better).
            pix_fmt: Pixel format (default 'yuv420p' for compatibility).

        Returns:
            Absolute path to the encoded output video.

        Raises:
            ValueError: If frames list is empty or fps is invalid.
            FileNotFoundError: If FFmpeg is not found.
            RuntimeError: If encoding fails.
        """
        if not frames:
            raise ValueError("Cannot encode: empty frames list.")
        if fps <= 0:
            raise ValueError(f"Invalid fps: {fps}")

        ffmpeg_path = self._find_ffmpeg()
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Encoding %d frames to %s (fps=%.2f, codec=%s, crf=%d)...",
            len(frames),
            output,
            fps,
            codec,
            crf,
        )

        # Create temp directory for frame PNGs within the output directory
        temp_dir = tempfile.mkdtemp(
            prefix="venc_frames_",
            dir=str(output.parent),
        )

        try:
            # Save frames as numbered PNGs
            logger.info("Saving %d frames as temporary PNGs...", len(frames))
            num_digits = len(str(len(frames)))
            frame_pattern = os.path.join(temp_dir, f"frame_%0{num_digits}d.png")

            for i, frame in enumerate(frames):
                frame_path = os.path.join(
                    temp_dir, f"frame_{i:0{num_digits}d}.png"
                )
                success = cv2.imwrite(frame_path, frame)
                if not success:
                    raise RuntimeError(f"Failed to save frame {i} to {frame_path}")

            # Build FFmpeg command for video encoding
            # Step 1: Encode video (without audio initially if we need to merge)
            if audio_source and Path(audio_source).is_file():
                video_only_path = str(output.with_suffix(".tmp.mp4"))
            else:
                video_only_path = str(output)

            encode_cmd = [
                ffmpeg_path,
                "-y",  # Overwrite output
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", codec,
                "-crf", str(crf),
                "-pix_fmt", pix_fmt,
                "-preset", "medium",
                "-movflags", "+faststart",
                video_only_path,
            ]

            logger.info("Running FFmpeg encode: %s", " ".join(encode_cmd))
            result = subprocess.run(
                encode_cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )

            if result.returncode != 0:
                logger.error("FFmpeg stderr: %s", result.stderr[-1000:])
                raise RuntimeError(
                    f"FFmpeg encoding failed with code {result.returncode}: "
                    f"{result.stderr[-500:]}"
                )

            logger.info("Video encoding complete: %s", video_only_path)

            # Step 2: Merge audio if source provided
            if audio_source and Path(audio_source).is_file():
                self._merge_audio(
                    ffmpeg_path,
                    video_only_path,
                    audio_source,
                    str(output),
                )
                # Clean up temporary video-only file
                try:
                    os.remove(video_only_path)
                except OSError:
                    pass

        finally:
            # Clean up temporary frame directory
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up temp frame directory: %s", temp_dir)
            except OSError as e:
                logger.warning("Failed to clean up temp directory %s: %s", temp_dir, e)

        final_path = str(output)
        file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
        logger.info(
            "Final video saved: %s (%.1f MB)", final_path, file_size_mb
        )

        return final_path

    @staticmethod
    def _merge_audio(
        ffmpeg_path: str,
        video_path: str,
        audio_source: str,
        output_path: str,
    ) -> None:
        """Merge audio from a source file into the encoded video.

        Args:
            ffmpeg_path: Path to FFmpeg executable.
            video_path: Path to the video-only encoded file.
            audio_source: Path to the original video/audio file.
            output_path: Path for the final output with audio.

        Raises:
            RuntimeError: If audio merging fails.
        """
        logger.info("Merging audio from %s...", audio_source)

        merge_cmd = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-i", audio_source,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]

        result = subprocess.run(
            merge_cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        if result.returncode != 0:
            logger.warning(
                "Audio merge failed (code %d), using video without audio. "
                "stderr: %s",
                result.returncode,
                result.stderr[-500:],
            )
            # Fall back: just copy the video-only file as the output
            shutil.copy2(video_path, output_path)
        else:
            logger.info("Audio merged successfully.")

    @staticmethod
    def _find_ffmpeg() -> str:
        """Locate the FFmpeg executable.

        Searches for FFmpeg in the system PATH using shutil.which. Also
        checks common installation locations on Windows.

        Returns:
            Path to the FFmpeg executable.

        Raises:
            FileNotFoundError: If FFmpeg is not found.
        """
        # Try system PATH first
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.debug("FFmpeg found at: %s", ffmpeg_path)
            return ffmpeg_path

        # Check common Windows locations
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
        ]

        for path in common_paths:
            if os.path.isfile(path):
                logger.debug("FFmpeg found at common path: %s", path)
                return path

        # Check FFMPEG_PATH environment variable
        env_path = os.environ.get("FFMPEG_PATH")
        if env_path and os.path.isfile(env_path):
            logger.debug("FFmpeg found via FFMPEG_PATH: %s", env_path)
            return env_path

        raise FileNotFoundError(
            "FFmpeg not found. Install FFmpeg and ensure it is in your PATH, "
            "or set the FFMPEG_PATH environment variable."
        )

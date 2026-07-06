import logging
import cv2
import numpy as np
from pathlib import Path
from scenedetect import detect, ContentDetector

logger = logging.getLogger(__name__)

class SceneDetector:
    def __init__(self, config):
        self.config = config

    def detect_scenes(self, video_path: str) -> list[tuple[int, int]]:
        """
        Detect scene boundaries and return a list of (start_frame, end_frame) tuples.
        """
        logger.info(f"Detecting scenes for {video_path}...")
        try:
            # Use PySceneDetect to find cuts
            scene_list = detect(video_path, ContentDetector(threshold=self.config.SCENE_THRESHOLD))
            scenes = []
            for scene in scene_list:
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()
                scenes.append((start_frame, end_frame))
            
            if not scenes:
                # Fallback to single scene of the entire video
                info = self.get_video_info(video_path)
                scenes = [(0, info['frame_count'] - 1)]
                
            logger.info(f"Detected {len(scenes)} scene(s).")
            return scenes
        except Exception as e:
            logger.error(f"Error during scene detection: {e}", exc_info=True)
            # Fallback to single scene
            info = self.get_video_info(video_path)
            return [(0, info['frame_count'] - 1)]

    def decode_video(self, video_path: str) -> tuple[list[np.ndarray], dict]:
        """
        Decode the entire video into a list of BGR frame arrays, and retrieve video metadata.
        """
        logger.info(f"Decoding video {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        
        frames = []
        info = self.get_video_info(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        logger.info(f"Successfully decoded {len(frames)} frames.")
        return frames, info

    def get_video_info(self, video_path: str) -> dict:
        """
        Retrieve video metadata.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'duration': duration
        }

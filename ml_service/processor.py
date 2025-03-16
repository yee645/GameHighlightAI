import cv2
import numpy as np
import torch
from torch.nn.functional import normalize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, frame_rate=30, max_frames=300):
        self.frame_rate = frame_rate
        self.max_frames = max_frames
        logger.info(f"Initialized VideoProcessor with frame_rate={frame_rate}, max_frames={max_frames}")

    def extract_video_features(self, video_path):
        logger.info(f"Starting video feature extraction from: {video_path}")
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Calculate frame sampling rate to match max_frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = max(1, total_frames // self.max_frames)
        logger.info(f"Total frames: {total_frames}, sampling every {sample_every} frames")

        frame_count = 0
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every nth frame
            if frame_count % sample_every == 0:
                try:
                    # Apply data augmentation
                    frame = self._augment_frame(frame)

                    # Preprocess frame
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).float() / 255.0
                    frame = frame.permute(2, 0, 1)  # HWC to CHW
                    frames.append(frame)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
                    raise

            frame_count += 1

        cap.release()
        logger.info(f"Processed {len(frames)} frames")

        if not frames:
            raise ValueError("No frames could be extracted from the video")

        # Pad if necessary
        while len(frames) < self.max_frames:
            frames.append(frames[-1])  # Repeat last frame

        frames = frames[:self.max_frames]  # Ensure exact length
        logger.info("Feature extraction completed successfully")
        return torch.stack(frames)

    def _augment_frame(self, frame):
        """Apply random augmentations to improve model robustness"""
        try:
            # Random brightness adjustment
            if np.random.random() < 0.5:
                brightness = 0.8 + np.random.random() * 0.4  # 0.8-1.2
                frame = cv2.convertScaleAbs(frame, alpha=brightness)

            # Random horizontal flip
            if np.random.random() < 0.5:
                frame = cv2.flip(frame, 1)

            return frame
        except Exception as e:
            logger.error(f"Error in frame augmentation: {str(e)}")
            raise

    def extract_audio_features(self, video_path):
        logger.info("Starting audio feature extraction")
        cap = cv2.VideoCapture(video_path)
        audio_features = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = max(1, total_frames // self.max_frames)

        frame_count = 0
        while cap.isOpened() and len(audio_features) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_every == 0:
                try:
                    # Convert frame to grayscale for audio proxy
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Calculate frame energy as audio proxy
                    energy = np.sum(gray * gray) / (gray.shape[0] * gray.shape[1])

                    # Calculate spectral centroid
                    f = np.fft.fft2(gray)
                    freq = np.fft.fftfreq(gray.shape[0])
                    centroid = np.sum(np.abs(f) * freq[:, np.newaxis]) / np.sum(np.abs(f))

                    features = np.array([energy, np.abs(centroid)])
                    audio_features.append(features)
                except Exception as e:
                    logger.error(f"Error processing audio frame {frame_count}: {str(e)}")
                    raise

            frame_count += 1

        cap.release()

        if not audio_features:
            raise ValueError("No audio features could be extracted")

        # Pad if necessary
        while len(audio_features) < self.max_frames:
            audio_features.append(audio_features[-1])

        audio_features = audio_features[:self.max_frames]

        # Normalize features and ensure consistent dimensionality
        audio_features = np.array(audio_features)
        mean = np.mean(audio_features, axis=0)
        std = np.std(audio_features, axis=0)
        audio_features = (audio_features - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

        # Reshape to match expected input size (N, 2) -> (N, 128)
        expanded_features = np.zeros((len(audio_features), 128))
        expanded_features[:, 0:2] = audio_features  # Keep original features in first positions

        audio_tensor = torch.from_numpy(expanded_features).float()
        logger.info("Audio feature extraction completed successfully")
        return audio_tensor
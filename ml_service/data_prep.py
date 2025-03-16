import os
import cv2
import numpy as np
import json
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparator:
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "annotations"), exist_ok=True)
        logger.info(f"Initialized DatasetPreparator with data_dir={data_dir}")

    def prepare_test_data(self) -> List[Tuple[str, List[float]]]:
        """Generate multiple test videos with different types of highlights"""
        video_paths = []
        highlight_timestamps_list = []

        # Generate multiple test videos with different patterns
        patterns = [
            {"name": "motion", "duration": 10, "highlights": [2.0, 5.0, 8.0]},
            {"name": "scene_change", "duration": 12, "highlights": [3.0, 6.0, 9.0]},
            {"name": "bright_moment", "duration": 8, "highlights": [2.5, 4.5, 6.5]},
        ]

        for pattern in patterns:
            video_path = os.path.join(self.data_dir, "videos", f"test_video_{pattern['name']}.mp4")
            logger.info(f"Creating test video at {video_path}")

            width, height = 640, 480
            fps = 30.0
            duration = pattern["duration"]
            highlight_timestamps = pattern["highlights"]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            logger.info(f"Setting highlight timestamps at: {highlight_timestamps}")

            for t in range(int(duration * fps)):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                current_time = t / fps

                # Create different types of highlights
                is_highlight = any(abs(current_time - ht) < 0.1 for ht in highlight_timestamps)

                if is_highlight:
                    if pattern["name"] == "motion":
                        # Create motion-based highlight
                        x = int(width/2 + width/4 * np.sin(current_time * 4))
                        cv2.circle(frame, (x, height//2), 40, (0, 0, 255), -1)
                        cv2.putText(frame, "MOTION", (width//3, height//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    elif pattern["name"] == "scene_change":
                        # Create scene change highlight
                        frame[:, :] = [0, 255, 0]
                        cv2.putText(frame, "SCENE CHANGE", (width//3, height//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    else:
                        # Create brightness-based highlight
                        frame[:, :] = [255, 255, 255]
                        cv2.putText(frame, "BRIGHT", (width//3, height//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                else:
                    # Normal frames with some basic motion
                    cv2.putText(frame, f"Time: {current_time:.1f}s", (width//4, height//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    circle_x = int(width/2 + width/4 * np.sin(current_time * 2))
                    cv2.circle(frame, (circle_x, height//2), 20, (0, 255, 0), -1)

                out.write(frame)

            out.release()
            logger.info(f"Test video generation completed for {pattern['name']}")

            # Save annotations
            annotation_path = os.path.join(self.data_dir, "annotations", f"test_video_{pattern['name']}.json")
            with open(annotation_path, 'w') as f:
                json.dump({
                    "highlights": highlight_timestamps,
                    "duration": duration,
                    "fps": fps,
                    "type": pattern["name"]
                }, f)
            logger.info(f"Saved annotations to {annotation_path}")

            video_paths.append(video_path)
            highlight_timestamps_list.append(highlight_timestamps)

            # Copy to test_videos directory for ML service
            os.makedirs("test_videos", exist_ok=True)
            test_video_path = os.path.join("test_videos", f"test_video_{pattern['name']}.mp4")
            import shutil
            shutil.copy2(video_path, test_video_path)
            logger.info(f"Copied test video to {test_video_path}")

        return list(zip(video_paths, highlight_timestamps_list))

    def validate_video(self, video_path: str) -> bool:
        """Validate that video can be opened and read"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, _ = cap.read()
            cap.release()
            return ret
        except Exception as e:
            logger.error(f"Video validation failed: {str(e)}")
            return False

    def prepare_dataset(self) -> List[Tuple[str, List[float]]]:
        """Prepare the synthetic test dataset"""
        logger.info("Preparing test data...")
        videos = self.prepare_test_data()

        # Validate videos
        valid_videos = []
        for video_path, timestamps in videos:
            if self.validate_video(video_path):
                valid_videos.append((video_path, timestamps))
                logger.info(f"Validated video: {video_path}")
            else:
                logger.warning(f"Skipping invalid video: {video_path}")

        logger.info(f"Prepared {len(valid_videos)} valid videos")
        return valid_videos

if __name__ == "__main__":
    preparator = DatasetPreparator()
    training_data = preparator.prepare_dataset()
    print(f"Prepared {len(training_data)} videos for training")
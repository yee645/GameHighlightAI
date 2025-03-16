import torch
import torch.nn as nn
import torch.optim as optim
from models import create_highlight_detection_model
from processor import VideoProcessor
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class HighlightTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = create_highlight_detection_model().to(device)
        self.processor = VideoProcessor()
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss for better numerical stability
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)  # Reduced learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        logger.info(f"Initialized trainer on device: {device}")

    def normalize_features(self, features):
        # Add feature normalization
        mean = features.mean(dim=(0, 1), keepdim=True)
        std = features.std(dim=(0, 1), keepdim=True) + 1e-8
        return (features - mean) / std

    def prepare_data(self, video_path: str, highlight_timestamps: List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_features = self.processor.extract_video_features(video_path)
        audio_features = self.processor.extract_audio_features(video_path)

        # Normalize features
        video_features = self.normalize_features(video_features)
        audio_features = self.normalize_features(audio_features)

        num_frames = len(video_features)
        labels = torch.zeros(num_frames)

        # Create smoother labels around highlights
        for timestamp in highlight_timestamps:
            frame_idx = int(timestamp * self.processor.frame_rate)
            if frame_idx < num_frames:
                window_size = 15
                for i in range(max(0, frame_idx - window_size), min(num_frames, frame_idx + window_size + 1)):
                    distance = abs(i - frame_idx)
                    weight = np.exp(-0.5 * (distance / 5) ** 2)
                    labels[i] = max(labels[i], weight)

        # Add small epsilon to prevent log(0)
        labels = labels.clamp(1e-7, 1 - 1e-7)

        video_features = video_features.unsqueeze(0)
        audio_features = audio_features.unsqueeze(0)
        labels = labels.unsqueeze(0)

        return (
            video_features.to(self.device),
            audio_features.to(self.device),
            labels.to(self.device)
        )

    def train(self, video_paths: List[str], highlight_timestamps_list: List[List[float]], epochs=50, batch_size=16):
        early_stopping = EarlyStopping(patience=5)
        best_accuracy = 0.0
        train_size = int(0.8 * len(video_paths))

        train_videos = video_paths[:train_size]
        train_timestamps = highlight_timestamps_list[:train_size]
        val_videos = video_paths[train_size:]
        val_timestamps = highlight_timestamps_list[train_size:]

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct_preds = 0
            total_preds = 0

            # Training phase
            for video_path, timestamps in zip(train_videos, train_timestamps):
                try:
                    video_features, audio_features, labels = self.prepare_data(video_path, timestamps)

                    self.optimizer.zero_grad()
                    logits = self.model(video_features, audio_features)
                    predictions = torch.sigmoid(logits)  # Apply sigmoid after model output

                    loss = self.criterion(logits, labels)

                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.error("NaN loss detected, skipping batch")
                        continue

                    loss.backward()

                    # Gradient clipping with lower threshold
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    # Calculate metrics
                    pred_labels = (predictions > 0.5).float()
                    correct_preds += (pred_labels == labels).sum().item()
                    total_preds += labels.numel()
                    total_loss += loss.item() * labels.numel()

                except Exception as e:
                    logger.error(f"Error during training: {str(e)}")
                    continue

            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_videos, val_timestamps)
            self.scheduler.step(val_loss)

            # Early stopping check
            early_stopping(val_loss)

            # Calculate and log metrics
            train_loss = total_loss / total_preds if total_preds > 0 else float('inf')
            train_accuracy = correct_preds / total_preds if total_preds > 0 else 0

            logger.info(f'Epoch {epoch + 1}/{epochs}:')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model('best_model.pth')
                logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")

            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break

            if val_accuracy >= 0.90:
                logger.info(f"Reached target accuracy of 90%")
                break

        self.load_model('best_model.pth')
        return best_accuracy

    def evaluate(self, video_paths: List[str], highlight_timestamps_list: List[List[float]]) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for video_path, timestamps in zip(video_paths, highlight_timestamps_list):
                try:
                    video_features, audio_features, labels = self.prepare_data(video_path, timestamps)
                    logits = self.model(video_features, audio_features)
                    predictions = torch.sigmoid(logits)

                    loss = self.criterion(logits, labels)

                    if not torch.isnan(loss):
                        pred_labels = (predictions > 0.5).float()
                        correct_preds += (pred_labels == labels).sum().item()
                        total_preds += labels.numel()
                        total_loss += loss.item() * labels.numel()
                except Exception as e:
                    logger.error(f"Error during evaluation: {str(e)}")
                    continue

        avg_loss = total_loss / total_preds if total_preds > 0 else float('inf')
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        return avg_loss, accuracy

    def predict(self, video_path: str) -> List[float]:
        logger.info(f"Starting prediction for video: {video_path}")
        self.model.eval()

        try:
            with torch.no_grad():
                logger.info("Extracting video features...")
                video_features = self.processor.extract_video_features(video_path)
                logger.info("Extracting audio features...")
                audio_features = self.processor.extract_audio_features(video_path)

                # Normalize features
                video_features = self.normalize_features(video_features)
                audio_features = self.normalize_features(audio_features)

                # Add batch dimension
                video_features = video_features.unsqueeze(0)
                audio_features = audio_features.unsqueeze(0)

                # Move to device
                video_features = video_features.to(self.device)
                audio_features = audio_features.to(self.device)

                # Get predictions
                logger.info("Running model inference...")
                logits = self.model(video_features, audio_features)
                predictions = torch.sigmoid(logits)
                predictions = predictions.squeeze(0)  # Remove batch dimension

                # Find highlight frames
                highlight_frames = torch.where(predictions > 0.5)[0].cpu().numpy()
                highlight_timestamps = highlight_frames / self.processor.frame_rate

                logger.info(f"Found {len(highlight_timestamps)} highlights")

                # Merge nearby highlights
                merged_highlights = []
                if len(highlight_timestamps) > 0:
                    current_start = highlight_timestamps[0]
                    current_end = current_start

                    for ts in highlight_timestamps[1:]:
                        if ts - current_end <= 1.0:  # Merge if less than 1 second apart
                            current_end = ts
                        else:
                            merged_highlights.append((current_start + current_end) / 2)
                            current_start = ts
                            current_end = ts

                    merged_highlights.append((current_start + current_end) / 2)

                logger.info(f"Merged into {len(merged_highlights)} final highlights")
                return merged_highlights

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise

    def save_model(self, path: str):
        logger.info(f"Saving model to {path}")
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        logger.info(f"Loading model from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
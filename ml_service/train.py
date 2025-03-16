import os
from trainer import HighlightTrainer
from data_prep import DatasetPreparator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_until_accuracy(target_accuracy=0.90, max_epochs=50):
    """Train model until target accuracy is reached, with better error handling and progress tracking"""
    # Prepare dataset
    preparator = DatasetPreparator()
    video_data = preparator.prepare_dataset()

    if not video_data:
        logger.error("No training data available")
        return False

    # Initialize trainer
    trainer = HighlightTrainer()

    # Train until target accuracy or max epochs reached
    try:
        accuracy = trainer.train(
            [v[0] for v in video_data], 
            [v[1] for v in video_data],
            epochs=max_epochs
        )

        logger.info(f"Final model accuracy: {accuracy:.4f}")

        if accuracy < target_accuracy:
            logger.warning("\nSuggestions for improving accuracy:")
            logger.warning("1. Increase training data variety")
            logger.warning("2. Try adjusting learning rate and batch size")
            logger.warning("3. Consider data augmentation techniques")
            logger.warning("4. Experiment with different model architectures")

        return accuracy >= target_accuracy

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return False

if __name__ == "__main__":
    # Create training directories if they don't exist
    os.makedirs("training_data/videos", exist_ok=True)
    os.makedirs("training_data/annotations", exist_ok=True)

    success = train_until_accuracy()
    if success:
        logger.info("Training completed successfully with target accuracy achieved")
    else:
        logger.error("Training completed but target accuracy not reached")
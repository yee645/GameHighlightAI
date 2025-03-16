import traceback
from flask import Flask, request, jsonify, send_from_directory
import tempfile
import os
import json
from trainer import HighlightTrainer
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, 
     origins=[r".*\.replit\.dev$", r".*\.repl\.co$"],
     supports_credentials=True)

trainer = HighlightTrainer()

# Create directories for storing test videos
UPLOAD_FOLDER = 'test_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model if exists
MODEL_PATH = "best_model.pth"
if os.path.exists(MODEL_PATH):
    trainer.load_model(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
else:
    logger.warning("No pre-trained model found")

# Store a test video for debugging
TEST_VIDEO_PATH = os.path.join(UPLOAD_FOLDER, "test_video.mp4")

@app.route('/api/train', methods=['POST'])
def train():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    try:
        timestamps = json.loads(request.form.get('timestamps', '[]'))
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid timestamps format'}), 400

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, 'video.mp4')
    video_file.save(video_path)

    try:
        trainer.train([video_path], [timestamps], epochs=50)
        trainer.save_model(MODEL_PATH)
        return jsonify({'message': 'Training completed successfully'})
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

@app.route('/api/test-video', methods=['GET'])
def get_test_video():
    """Endpoint to serve test video for debugging"""
    if not os.path.exists(TEST_VIDEO_PATH):
        return jsonify({'error': 'No test video available'}), 404
    logger.info(f"Serving test video from: {TEST_VIDEO_PATH}")
    return send_from_directory(os.path.dirname(TEST_VIDEO_PATH), 
                             os.path.basename(TEST_VIDEO_PATH), 
                             as_attachment=True)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, 'video.mp4')
    video_file.save(video_path)

    try:
        # Store copy for testing if it's new
        if not os.path.exists(TEST_VIDEO_PATH):
            video_file.save(TEST_VIDEO_PATH)
            logger.info(f"Saved test video to {TEST_VIDEO_PATH}")

        # Get predictions
        logger.info(f"Processing video for predictions: {video_path}")
        timestamps = trainer.predict(video_path)
        logger.info(f"Generated predictions: {timestamps}")

        highlights = [
            {
                'timestamp': float(ts),
                'type': 'ml',
                'confidence': 0.9
            } for ts in timestamps
        ]
        logger.info(f"Formatted highlights: {highlights}")

        return jsonify({
            'highlights': highlights
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

if __name__ == '__main__':
    # Train initial model if needed
    if not os.path.exists(MODEL_PATH):
        from data_prep import DatasetPreparator
        preparator = DatasetPreparator()
        training_data = preparator.prepare_dataset()
        if training_data:
            logger.info("Training initial model...")
            trainer.train([v[0] for v in training_data], 
                        [v[1] for v in training_data],
                        epochs=50)
            trainer.save_model(MODEL_PATH)
            logger.info("Initial model training completed")

    app.run(host='0.0.0.0', port=5001, debug=True)
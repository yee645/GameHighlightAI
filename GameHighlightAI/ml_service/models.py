import torch
import torch.nn as nn

class HighlightDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Efficient video feature extraction
        self.video_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Enhanced audio processing
        self.audio_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        # Multi-scale temporal convolution
        self.temporal_conv = nn.ModuleList([
            nn.Conv1d(512, 128, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])

        # Lightweight LSTM
        self.lstm = nn.LSTM(
            input_size=384,  # 128 * 3 from temporal convolution
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 from bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, video_frames, audio_features):
        batch_size, time_steps = video_frames.shape[:2]

        # Process video features
        video_features = []
        for t in range(time_steps):
            frame = video_frames[:, t]
            features = self.video_cnn(frame)
            video_features.append(features.squeeze(-1).squeeze(-1))
        video_features = torch.stack(video_features, dim=1)  # [B, T, 256]

        # Process audio features
        audio_features_processed = []
        for t in range(time_steps):
            audio_t = audio_features[:, t]
            audio_t = self.audio_net(audio_t)
            audio_features_processed.append(audio_t)
        audio_features = torch.stack(audio_features_processed, dim=1)  # [B, T, 256]

        # Combine features
        combined = torch.cat([video_features, audio_features], dim=2)  # [B, T, 512]

        # Apply self-attention
        attn_out = combined.transpose(0, 1)  # [T, B, 512]
        attn_out, _ = self.attention(attn_out, attn_out, attn_out)
        attn_out = attn_out.transpose(0, 1)  # [B, T, 512]

        # Multi-scale temporal processing
        temp_features = attn_out.transpose(1, 2)  # [B, 512, T]
        multi_scale = []
        for conv in self.temporal_conv:
            out = conv(temp_features)
            multi_scale.append(out)
        multi_scale = torch.cat(multi_scale, dim=1)  # [B, 384, T]
        multi_scale = multi_scale.transpose(1, 2)  # [B, T, 384]

        # LSTM processing
        lstm_out, _ = self.lstm(multi_scale)  # [B, T, 512]

        # Generate predictions
        predictions = []
        for t in range(time_steps):
            pred = self.classifier(lstm_out[:, t])
            predictions.append(pred)

        return torch.cat(predictions, dim=1)  # [B, T]

def create_highlight_detection_model():
    return HighlightDetectionModel()
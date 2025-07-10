# models/model.py
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # Sequence model
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)  # Projection layer

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B, L, hidden_dim]
        pooled = lstm_out.mean(dim=1)  # [B, hidden_dim]
        embedding = self.embedding_layer(pooled)  # [B, embedding_dim]
        return embedding  # Return sensor embedding

class MultimodalOdorNet(nn.Module):
    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen-VL-Chat-Int4",
        sensor_feat_dim: int = 8,
        sensor_hidden_dim: int = 128,
        sensor_emb_dim: int = 768,
        fusion_dim: int = 768,
        num_perfumes: int = 12
    ):
        super().__init__()
        # Load Qwen processor and model for image+text
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_model_name)
        self.qwen_model = AutoModel.from_pretrained(qwen_model_name)

        # Sensor sequence encoder
        self.smell_enc = LSTMNet(sensor_feat_dim, sensor_hidden_dim, sensor_emb_dim)
        self.smell_proj = nn.Linear(sensor_emb_dim, fusion_dim)  # Project sensor embedding

        # Projection layers for vision and text features
        v_dim = self.qwen_model.config.vision_hidden_size
        t_dim = self.qwen_model.config.hidden_size
        self.vision_proj = nn.Linear(v_dim, fusion_dim)
        self.text_proj = nn.Linear(t_dim, fusion_dim)

        # Transformer-based fusion module
        layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=4)
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)

        # Output head for mixture ratio prediction
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, num_perfumes),
            nn.Softmax(dim=-1)
        )

    def forward(self, image, text, smell_input):
        # Prepare image+text for Qwen
        inputs = self.qwen_processor(text=text, images=image, return_tensors="pt", padding=True)
        # Move inputs to the same device as smell_input
        inputs = {k: v.to(smell_input.device) for k, v in inputs.items()}
        qwen_out = self.qwen_model(**inputs, output_hidden_states=True)

        vision_feat = qwen_out.vision_hidden_state.mean(dim=1)  # Visual embedding
        text_feat   = qwen_out.last_hidden_state.mean(dim=1)    # Textual embedding
        smell_emb   = self.smell_enc(smell_input)              # Sensor embedding

        # Project each modality into fusion space
        v_proj = self.vision_proj(vision_feat)
        t_proj = self.text_proj(text_feat)
        s_proj = self.smell_proj(smell_emb)

        # Stack and fuse via transformer
        fused = torch.stack([v_proj, t_proj, s_proj], dim=1)  # [B,3,fusion_dim]
        fused_encoded = self.fusion(fused)                     # [B,3,fusion_dim]
        fused_summary = fused_encoded.mean(dim=1)              # [B,fusion_dim]

        return self.output_head(fused_summary)  # Predict mixture ratios

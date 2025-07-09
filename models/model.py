import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq, hidden)
        pooled = torch.mean(lstm_out, dim=1)  # average over time
        embedding = self.embedding_layer(pooled)
        out = self.classifier(embedding)
        return out, embedding


class SmellReconstructionModel(nn.Module):
    def __init__(
        self,
        qwen_model_name="Qwen/Qwen-VL-Chat-Int4",
        smellnet_model=None,
        fusion_dim=768,
        num_perfumes=12
    ):
        super().__init__()
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_model_name)
        self.qwen_model = AutoModel.from_pretrained(qwen_model_name)

        self.smellnet_model = smellnet_model  # Your pretrained encoder for sensor input
        self.smell_proj = nn.Linear(smellnet_model.output_dim, fusion_dim)

        self.vision_proj = nn.Linear(self.qwen_model.config.vision_hidden_size, fusion_dim)
        self.text_proj = nn.Linear(self.qwen_model.config.hidden_size, fusion_dim)

        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=4),
            num_layers=2
        )

        # Predict proportions over 12 base perfumes (normalized)
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, num_perfumes),
            nn.Softmax(dim=-1)  # mixture ratios
        )

    def forward(self, image, text, smell_input):
        """
        Args:
            image (PIL.Image or tensor): Image of the food
            text (str): Text description
            smell_input (tensor): Sensor vector or sequence

        Returns:
            torch.Tensor: [B, 12] mixture ratio over perfume bases
        """
        inputs = self.qwen_processor(text=text, images=image, return_tensors="pt", padding=True).to(image.device)
        qwen_outputs = self.qwen_model(**inputs, output_hidden_states=True)

        vision_feat = qwen_outputs.vision_hidden_state.mean(dim=1)  # [B, V_dim]
        text_feat = qwen_outputs.last_hidden_state.mean(dim=1)      # [B, T_dim]
        smell_feat = self.smellnet_model(smell_input)               # [B, S_dim]

        vision_proj = self.vision_proj(vision_feat)
        text_proj = self.text_proj(text_feat)
        smell_proj = self.smell_proj(smell_feat)

        fused = torch.stack([vision_proj, text_proj, smell_proj], dim=1)  # [B, 3, fusion_dim]
        fused_encoded = self.fusion(fused)                                # [B, 3, fusion_dim]
        fused_summary = fused_encoded.mean(dim=1)                         # [B, fusion_dim]

        return self.output_head(fused_summary)                            # [B, 12]

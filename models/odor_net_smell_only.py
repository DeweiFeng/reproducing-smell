import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    """
    Single-layer LSTM encoder producing a fixed embedding via mean-pooling.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, input_dim]
        out, _ = self.lstm(x)            # out: [B, L, hidden_dim]
        pooled = out.mean(dim=1)         # pooled: [B, hidden_dim]
        return self.embedding_layer(pooled)  # [B, embedding_dim]


class StackedLSTMNet(nn.Module):
    """
    Multi-layer LSTM encoder with dropout, producing an embedding via mean-pooling.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, input_dim]
        out, _ = self.lstm(x)            # out: [B, L, hidden_dim]
        pooled = out.mean(dim=1)         # pooled: [B, hidden_dim]
        return self.embedding_layer(pooled)  # [B, embedding_dim]


class OdorNet(nn.Module):
    """
    OdorNet model for classifying smell sequences into perfume categories.

    Uses a stacked LSTM encoder for richer feature extraction, followed by a projection
    layer and output classification head.
    """
    def __init__(
        self,
        sensor_feat_dim: int = 12,
        sensor_hidden_dim: int = 128,
        sensor_emb_dim: int = 768,
        fusion_dim: int = 768,
        num_perfumes: int = 12,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1
    ):
        super().__init__()
        # Choose between single vs stacked LSTM:
        # self.smell_enc = LSTMNet(sensor_feat_dim, sensor_hidden_dim, sensor_emb_dim)
        self.smell_enc = StackedLSTMNet(
            sensor_feat_dim,
            sensor_hidden_dim,
            sensor_emb_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout
        )
        # Projection to fusion space
        self.smell_proj = nn.Linear(sensor_emb_dim, fusion_dim)
        # Classification head
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, num_perfumes)
        )

    def forward(self, smell_input: torch.Tensor) -> torch.Tensor:
        # smell_input: [B, L, sensor_feat_dim]
        emb = self.smell_enc(smell_input)   # [B, sensor_emb_dim]
        proj = self.smell_proj(emb)         # [B, fusion_dim]
        return self.output_head(proj)       # [B, num_perfumes]

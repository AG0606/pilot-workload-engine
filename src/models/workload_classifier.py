from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from .cardio_encoder import CardioEncoder
from .context_encoder import FlightContextEncoder
from .eeg_encoder import EEGNetEncoder
from .fusion_network import CrossModalAttentionFusion
from .ocular_encoder import OcularEncoder


class MultiModalWorkloadClassifier(nn.Module):
    """End-to-end multi-modal deep network for real-time pilot cognitive state classification."""

    def __init__(
        self,
        num_classes: int = 4,
        eeg_channels: int = 17,
        cardio_channels: int = 3,
        ocular_channels: int = 4,
        context_channels: int = 8,
        temporal_samples: int = 80,
        embed_dim: int = 128,
        fused_dim: int = 256,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Modality encoders
        self.eeg_encoder = EEGNetEncoder(
            num_channels=eeg_channels,
            temporal_samples=temporal_samples,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate,
        )
        self.cardio_encoder = CardioEncoder(
            in_channels=cardio_channels,
            embed_dim=embed_dim,
        )
        self.ocular_encoder = OcularEncoder(
            in_channels=ocular_channels,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate,
        )
        self.context_encoder = FlightContextEncoder(
            in_channels=context_channels,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate,
        )

        # Attention fusion
        self.fusion = CrossModalAttentionFusion(
            embed_dim=embed_dim,
            num_modalities=4,
            fused_dim=fused_dim,
            dropout_rate=dropout_rate,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(
        self,
        eeg: Optional[torch.Tensor] = None,
        cardio: Optional[torch.Tensor] = None,
        ocular: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Processes either explicit modality tensors or a batch dictionary."""
        if batch is not None:
            eeg = batch["eeg"]
            cardio = batch["cardio"]
            ocular = batch["ocular"]
            context = batch["context"]

        if eeg is None or cardio is None or ocular is None or context is None:
            raise ValueError("All 4 modalities (eeg, cardio, ocular, context) must be provided.")

        eeg_emb = self.eeg_encoder(eeg)
        cardio_emb = self.cardio_encoder(cardio)
        ocular_emb = self.ocular_encoder(ocular)
        context_emb = self.context_encoder(context)

        fused = self.fusion([eeg_emb, cardio_emb, ocular_emb, context_emb])
        logits = self.classifier(fused)
        return logits

    def predict_proba(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Computes softmax probabilities across the 4 cognitive states."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch=batch)
            probs = torch.softmax(logits, dim=-1)
        return probs

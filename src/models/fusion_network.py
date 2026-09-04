from typing import List
import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """Multi-head cross-modal attention fusing neurological, autonomic, ocular, and flight dynamics."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_modalities: int = 4,
        num_heads: int = 4,
        fused_dim: int = 256,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Learnable modality tokens
        self.modality_embeddings = nn.Parameter(torch.randn(1, num_modalities, embed_dim) * 0.02)

        # Multi-Head Attention layer across modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward refinement block
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Output projection from flattened modality tokens
        self.fusion_projection = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, modality_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Expects list of tensors each of shape [B, embed_dim]."""
        if len(modality_embeddings) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(modality_embeddings)}"
            )

        # Stack into [B, num_modalities, embed_dim]
        stacked = torch.stack(modality_embeddings, dim=1)
        # Add modality identity tokens
        x = stacked + self.modality_embeddings

        # Self-attention across modalities
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Flatten all modality tokens and project to fused representation
        flattened = x.view(x.size(0), -1)
        fused = self.fusion_projection(flattened)
        return fused

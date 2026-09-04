import unittest
import torch
from src.models.cardio_encoder import CardioEncoder
from src.models.context_encoder import FlightContextEncoder
from src.models.eeg_encoder import EEGNetEncoder
from src.models.fusion_network import CrossModalAttentionFusion
from src.models.ocular_encoder import OcularEncoder
from src.models.workload_classifier import MultiModalWorkloadClassifier


class TestModels(unittest.TestCase):
    """Unit tests verifying neural encoder forward passes, fusion output dimensions, and gradient flow."""

    def setUp(self) -> None:
        self.batch_size = 4
        self.temporal_samples = 80
        self.embed_dim = 128
        self.fused_dim = 256

        self.eeg = torch.randn(self.batch_size, 17, self.temporal_samples)
        self.cardio = torch.randn(self.batch_size, 3, self.temporal_samples)
        self.ocular = torch.randn(self.batch_size, 4, self.temporal_samples)
        self.context = torch.randn(self.batch_size, 8, self.temporal_samples)
        self.labels = torch.randint(0, 4, (self.batch_size,), dtype=torch.int64)

    def test_eegnet_encoder(self) -> None:
        """Verifies EEGNet output shape matches expected embedding dimension."""
        model = EEGNetEncoder(num_channels=17, temporal_samples=self.temporal_samples, embed_dim=self.embed_dim)
        out = model(self.eeg)
        self.assertEqual(out.shape, (self.batch_size, self.embed_dim))

    def test_cardio_encoder(self) -> None:
        """Verifies Cardio residual encoder output shape."""
        model = CardioEncoder(in_channels=3, embed_dim=self.embed_dim)
        out = model(self.cardio)
        self.assertEqual(out.shape, (self.batch_size, self.embed_dim))

    def test_ocular_encoder(self) -> None:
        """Verifies Ocular encoder output shape."""
        model = OcularEncoder(in_channels=4, embed_dim=self.embed_dim)
        out = model(self.ocular)
        self.assertEqual(out.shape, (self.batch_size, self.embed_dim))

    def test_context_encoder(self) -> None:
        """Verifies Flight Context dynamics encoder output shape."""
        model = FlightContextEncoder(in_channels=8, embed_dim=self.embed_dim)
        out = model(self.context)
        self.assertEqual(out.shape, (self.batch_size, self.embed_dim))

    def test_attention_fusion(self) -> None:
        """Verifies multi-head attention fusion over 4 modality tokens."""
        fusion = CrossModalAttentionFusion(embed_dim=self.embed_dim, num_modalities=4, fused_dim=self.fused_dim)
        tokens = [torch.randn(self.batch_size, self.embed_dim) for _ in range(4)]
        fused = fusion(tokens)
        self.assertEqual(fused.shape, (self.batch_size, self.fused_dim))

    def test_end_to_end_classifier_and_gradients(self) -> None:
        """Verifies full multi-modal network forward output, backward gradients, and probability predictions."""
        model = MultiModalWorkloadClassifier(
            num_classes=4,
            eeg_channels=17,
            cardio_channels=3,
            ocular_channels=4,
            context_channels=8,
            temporal_samples=self.temporal_samples,
            embed_dim=self.embed_dim,
            fused_dim=self.fused_dim,
        )

        batch = {
            "eeg": self.eeg,
            "cardio": self.cardio,
            "ocular": self.ocular,
            "context": self.context,
            "label": self.labels,
        }

        logits = model(batch=batch)
        self.assertEqual(logits.shape, (self.batch_size, 4))

        # Backward gradient flow check
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient.")

        # Test predict_proba output distribution
        probs = model.predict_proba(batch)
        self.assertEqual(probs.shape, (self.batch_size, 4))
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(self.batch_size))


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.workload_dataset import PilotWorkloadDataset
from src.models.workload_classifier import MultiModalWorkloadClassifier
from src.training.losses import FocalLoss, compute_class_weights
from src.training.metrics import compute_classification_metrics, compute_kaggle_log_loss
from src.training.trainer import WorkloadTrainer


class TestLossesAndTraining(unittest.TestCase):
    """Unit tests for FocalLoss, Kaggle log loss metric, and training loop execution."""

    def setUp(self) -> None:
        self.num_classes = 4
        self.batch_size = 8
        self.logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        self.targets = torch.tensor([0, 0, 0, 1, 1, 2, 3, 0], dtype=torch.int64)

    def test_class_weights_computation(self) -> None:
        """Verifies inverse frequency class weights computation."""
        weights = compute_class_weights(self.targets, num_classes=self.num_classes)
        self.assertEqual(len(weights), self.num_classes)
        # Class 0 is most frequent, so weight should be lowest
        self.assertLess(weights[0].item(), weights[2].item())
        self.assertLess(weights[0].item(), weights[3].item())

    def test_focal_loss_forward_and_backward(self) -> None:
        """Verifies FocalLoss computes finite scalar and produces valid gradients."""
        weights = compute_class_weights(self.targets, num_classes=self.num_classes)
        criterion = FocalLoss(gamma=2.0, alpha=weights)

        loss = criterion(self.logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(self.logits.grad)
        self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_kaggle_log_loss_and_metrics(self) -> None:
        """Verifies Kaggle log loss metric and classification summary."""
        probs = torch.softmax(self.logits.detach(), dim=-1).numpy()
        targets_np = self.targets.numpy()

        loss_val = compute_kaggle_log_loss(targets_np, probs)
        self.assertGreater(loss_val, 0.0)

        metrics = compute_classification_metrics(targets_np, probs, num_classes=self.num_classes)
        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("kaggle_log_loss", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_trainer_single_epoch(self) -> None:
        """Verifies WorkloadTrainer executes training and evaluation without exceptions."""
        windows = {
            "eeg": np.random.randn(16, 17, 80).astype(np.float32),
            "cardio": np.random.randn(16, 3, 80).astype(np.float32),
            "ocular": np.random.randn(16, 4, 80).astype(np.float32),
            "context": np.random.randn(16, 8, 80).astype(np.float32),
            "label": np.random.randint(0, 4, size=16, dtype=np.int64),
        }
        dataset = PilotWorkloadDataset.from_window_dict(windows)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        model = MultiModalWorkloadClassifier(
            num_classes=4,
            eeg_channels=17,
            cardio_channels=3,
            ocular_channels=4,
            context_channels=8,
            temporal_samples=80,
            embed_dim=32,
            fused_dim=64,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = FocalLoss(gamma=2.0)
        device = torch.device("cpu")

        trainer = WorkloadTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        train_res = trainer.train_epoch(loader)
        self.assertIn("train_loss", train_res)
        self.assertGreater(train_res["train_loss"], 0.0)

        eval_res = trainer.evaluate(loader)
        self.assertIn("val_loss", eval_res)
        self.assertIn("kaggle_log_loss", eval_res)


if __name__ == "__main__":
    unittest.main()

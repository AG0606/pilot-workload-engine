import unittest
import torch

from src.models.ocular_vision import OcularVigilanceNet, PERCLOSCalculator, VigilanceAlertLevel


class TestOcularVisionAndPERCLOS(unittest.TestCase):
    """Unit tests for the OcularVigilanceNet CNN and PERCLOSCalculator."""

    def test_ocular_net_forward_and_gradients(self) -> None:
        """Verifies forward pass output shape and backward gradient calculation."""
        model = OcularVigilanceNet(in_channels=1, num_classes=2)
        batch_size = 4
        # Simulated grayscale eye crop [B, 1, 64, 64]
        x = torch.randn(batch_size, 1, 64, 64)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        logits = model(x)
        self.assertEqual(logits.shape, (batch_size, 2))

        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_ocular_net_predict_state(self) -> None:
        """Verifies single-image state inference helper."""
        model = OcularVigilanceNet(in_channels=1, num_classes=2)
        single_image = torch.randn(1, 1, 64, 64)
        pred_class, prob = model.predict_state(single_image)
        self.assertIn(pred_class, [0, 1])
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_perclos_nominal_condition(self) -> None:
        """Verifies nominal alert level when eyes are predominantly open."""
        calc = PERCLOSCalculator(window_duration_sec=10.0, fps=10.0)
        # Feed 100 open frames
        for _ in range(100):
            res = calc.update(is_closed=False)

        self.assertEqual(res["alert_level"], VigilanceAlertLevel.NOMINAL.value)
        self.assertAlmostEqual(res["perclos_percent"], 0.0)
        self.assertFalse(res["microsleep_detected"])

    def test_perclos_drowsiness_and_microsleep(self) -> None:
        """Verifies detection of microsleep and critical fatigue alarm."""
        calc = PERCLOSCalculator(
            window_duration_sec=5.0,
            fps=10.0,
            warning_perclos_threshold=40.0,
            critical_perclos_threshold=70.0,
            microsleep_duration_sec=1.5,
        )

        # Feed open eyes
        for _ in range(20):
            calc.update(is_closed=False)

        # Feed sustained closed eyes (microsleep: 20 frames = 2.0 sec > 1.5 sec)
        for _ in range(20):
            res = calc.update(is_closed=True)

        self.assertTrue(res["microsleep_detected"])
        self.assertEqual(res["alert_level"], VigilanceAlertLevel.FATIGUE_CRITICAL.value)
        self.assertGreater(res["perclos_percent"], 40.0)

    def test_blink_rate_counting(self) -> None:
        """Verifies counting of natural blinks lasting 100ms - 400ms."""
        calc = PERCLOSCalculator(window_duration_sec=10.0, fps=10.0)
        # 1 blink: 2 closed frames (200 ms @ 10 fps) then open
        calc.update(is_closed=False)
        calc.update(is_closed=True)
        calc.update(is_closed=True)
        res = calc.update(is_closed=False)

        self.assertEqual(res["blink_count"], 1)


if __name__ == "__main__":
    unittest.main()

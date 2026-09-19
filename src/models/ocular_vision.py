from collections import deque
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VigilanceAlertLevel(str, Enum):
    NOMINAL = "NOMINAL"
    DROWSINESS_WARNING = "DROWSINESS_WARNING"
    FATIGUE_CRITICAL = "FATIGUE_CRITICAL"


class OcularVigilanceNet(nn.Module):
    """Lightweight 2D Convolutional Neural Network for Eye State Classification (Open vs Closed).

    Designed for real-time edge avionics camera processing (Driver Drowsiness Dataset DDD / Webcam):
    - Input: Grayscale eye/face crop [B, 1, 64, 64]
    - Output: Logits for 2 classes [0: Open / Vigilant, 1: Closed / Drowsy]
    - Latency: < 1.0 ms on CPU
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),          # 16 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        logits = self.classifier(feat)
        return logits

    def predict_state(self, x: torch.Tensor) -> Tuple[int, float]:
        """Infers eye state and closure probability for a single frame."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            pred_class = int(torch.argmax(probs, dim=-1).item())
            closed_prob = float(probs[0, 1].item())
        return pred_class, closed_prob


class PERCLOSCalculator:
    """Sliding-window PERCLOS (% Eye Closure) and Blink Dynamics Engine.

    Implements Section 3 & 7 of the e-Pilot DBMS specification:
    - Tracks eye closure percentage over sliding window (e.g., 60 seconds).
    - Detects voluntary blinks (100 - 400 ms) vs micro-sleeps (> 1500 ms).
    - Emits real-time vigilance alert states for cockpit display integration.
    """

    def __init__(
        self,
        window_duration_sec: float = 60.0,
        fps: float = 30.0,
        warning_perclos_threshold: float = 40.0,
        critical_perclos_threshold: float = 80.0,
        microsleep_duration_sec: float = 1.5,
    ) -> None:
        self.window_duration_sec = window_duration_sec
        self.fps = fps
        self.capacity = int(round(window_duration_sec * fps))

        self.warning_threshold = warning_perclos_threshold
        self.critical_threshold = critical_perclos_threshold
        self.microsleep_frames = int(round(microsleep_duration_sec * fps))

        # Sliding buffers
        self.state_buffer: Deque[int] = deque(maxlen=self.capacity)
        self.consecutive_closed_frames: int = 0
        self.total_blinks: int = 0
        self.current_blink_frames: int = 0

    def update(self, is_closed: bool) -> Dict[str, Union[float, int, str, bool]]:
        """Processes a single frame eye-closure observation and updates metrics."""
        val = 1 if is_closed else 0
        self.state_buffer.append(val)

        # Micro-sleep and blink detection logic
        if is_closed:
            self.consecutive_closed_frames += 1
            self.current_blink_frames += 1
        else:
            # End of closure: check if it was a normal blink (100ms - 400ms = 3 - 12 frames @ 30fps)
            min_blink = int(0.10 * self.fps)
            max_blink = int(0.40 * self.fps)
            if min_blink <= self.current_blink_frames <= max_blink:
                self.total_blinks += 1
            self.current_blink_frames = 0
            self.consecutive_closed_frames = 0

        # Calculate current PERCLOS
        if len(self.state_buffer) == 0:
            perclos = 0.0
        else:
            perclos = float((sum(self.state_buffer) / len(self.state_buffer)) * 100.0)

        # Micro-sleep flag
        microsleep_detected = self.consecutive_closed_frames >= self.microsleep_frames

        # Alert level determination
        if perclos >= self.critical_threshold or microsleep_detected:
            alert = VigilanceAlertLevel.FATIGUE_CRITICAL
        elif perclos >= self.warning_threshold:
            alert = VigilanceAlertLevel.DROWSINESS_WARNING
        else:
            alert = VigilanceAlertLevel.NOMINAL

        # Blink rate (blinks per minute)
        buffered_sec = len(self.state_buffer) / self.fps if self.fps > 0 else 1.0
        blink_rate_bpm = float((self.total_blinks / max(buffered_sec, 1.0)) * 60.0)

        return {
            "perclos_percent": round(perclos, 2),
            "alert_level": alert.value,
            "microsleep_detected": microsleep_detected,
            "blink_count": self.total_blinks,
            "blink_rate_bpm": round(blink_rate_bpm, 1),
            "consecutive_closed_sec": round(self.consecutive_closed_frames / self.fps, 2),
        }

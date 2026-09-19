from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.mixture import GaussianMixture


class FlightPhase(str, Enum):
    TAXI_GROUND = "TAXI_GROUND"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    DESCENT = "DESCENT"
    EMERGENCY_UPSET = "EMERGENCY_UPSET"


@dataclass
class FlightTelemetrySample:
    altitude_ft: float
    vertical_speed_fpm: float
    indicated_airspeed_kts: float
    pitch_deg: float
    roll_deg: float
    vertical_g: float
    yaw_rate_deg_s: float = 0.0


class FlightPhaseSegmenter:
    """Classifies flight regimes into operational flight phases and emergency upset states.

    Directly implements Section 5 & 6.5 of the e-Pilot DBMS specification:
    - Combines kinematic rule-based avionics state machine with GMM statistical clustering.
    - Tags every window with flight_phase and is_emergency_simulated for SQL GROUP BY analysis.
    """

    def __init__(
        self,
        gmm_components: int = 4,
        random_state: int = 42,
    ) -> None:
        self.gmm = GaussianMixture(
            n_components=gmm_components,
            covariance_type="full",
            random_state=random_state,
        )
        self.is_gmm_fitted = False

    def classify_rule_based(self, telemetry: FlightTelemetrySample) -> Tuple[FlightPhase, bool]:
        """Classifies a telemetry point using deterministic aerodynamics thresholding."""
        # 1. Emergency upset condition takes highest priority
        is_emergency = (
            abs(telemetry.roll_deg) > 45.0
            or telemetry.pitch_deg > 25.0
            or telemetry.pitch_deg < -15.0
            or telemetry.vertical_g > 2.5
            or telemetry.vertical_g < 0.2
        )
        if is_emergency:
            return FlightPhase.EMERGENCY_UPSET, True

        # 2. Taxi / Ground
        if telemetry.altitude_ft < 150.0 and telemetry.indicated_airspeed_kts < 60.0:
            return FlightPhase.TAXI_GROUND, False

        # 3. Climb Phase
        if telemetry.vertical_speed_fpm > 400.0:
            return FlightPhase.CLIMB, False

        # 4. Descent / Approach
        if telemetry.vertical_speed_fpm < -400.0:
            return FlightPhase.DESCENT, False

        # 5. Cruise Phase
        return FlightPhase.CRUISE, False

    def segment_trajectory(
        self,
        telemetry_array: np.ndarray,  # Shape: [N, 6] -> [alt, vs, ias, pitch, roll, g_z]
    ) -> List[Tuple[FlightPhase, bool]]:
        """Segments a continuous flight trajectory into phase tags and emergency flags."""
        results: List[Tuple[FlightPhase, bool]] = []
        for row in telemetry_array:
            sample = FlightTelemetrySample(
                altitude_ft=float(row[0]),
                vertical_speed_fpm=float(row[1]),
                indicated_airspeed_kts=float(row[2]),
                pitch_deg=float(row[3]),
                roll_deg=float(row[4]),
                vertical_g=float(row[5]),
            )
            phase, is_emerg = self.classify_rule_based(sample)
            results.append((phase, is_emerg))
        return results

    def fit_clustering_model(self, kinematics_matrix: np.ndarray) -> None:
        """Fits Gaussian Mixture Model to identify latent aerodynamic regimes."""
        if len(kinematics_matrix) < 10:
            return
        self.gmm.fit(kinematics_matrix)
        self.is_gmm_fitted = True

    def predict_cluster(self, kinematics_matrix: np.ndarray) -> np.ndarray:
        """Predicts latent aerodynamic cluster IDs."""
        if not self.is_gmm_fitted:
            self.fit_clustering_model(kinematics_matrix)
        return self.gmm.predict(kinematics_matrix)

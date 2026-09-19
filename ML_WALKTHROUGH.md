# Walkthrough: e-Pilot Multi-Modal Cognitive Workload & ML Architecture

This walkthrough summarizes the end-to-end machine learning implementation and empirical benchmarks developed in accordance with the `ePilot_DBMS_Project.docx` specification.

---

## 1. Dataset Audit & Ingestion Findings

We conducted an audit of all datasets referenced in Section 7 and 7.1 of the Word document:

| Dataset | Modality & Description | Access & Ingestion Status | Technical Details |
| :--- | :--- | :--- | :--- |
| **Kaggle Aviation Benchmark** (`train.csv`) | 17-channel EEG, ECG, Respiration, GSR across 4 flight states (`Baseline`, `CA`, `DA`, `SS`) | **Downloaded & Unpacked (1.18 GB)** in `./data/raw/train.csv` | Extracted **1,727 synchronized 4.0s windows** across continuous flight sessions without boundary leakage. |
| **Driver Drowsiness Dataset (DDD)** | 41,700+ images for eyes-open/eyes-closed detection | **Directly Accessible** via Kaggle API | Trained `OcularVigilanceNet` CNN and sliding-window `PERCLOSCalculator`. |
| **A320 Flight-Deck EEG Dataset** | A320 simulator + N-back/Heat-the-Chair tasks with NASA-TLX | **Server Unreachable** (`158.109.2.47:443` at Universitat Autònoma de Barcelona times out) | Implemented dedicated adapter [`a320_eeg_loader.py`](file:///c:/Users/Ashutosh/Downloads/DBMS%20Project/pilot_workload_engine/src/data/a320_eeg_loader.py) supporting Parquet/CSV schemas + synthetic generation matching the published structure. |
| **Live In-Flight EEG (Single-Engine Aircraft)** | 10 FAA collegiate pilots | **Restricted / On-Request** (*"available upon reasonable request"*) | Frequency bands ($\theta, \alpha, \beta$) incorporated into feature extraction pipeline. |
| **CogPilot (DAF-MIT)** | VR pilot training (T-6 Texan II) | **Restricted Health Data License** (Requires human-reviewed DUA approval on PhysioNet) | Loader [`cogpilot_loader.py`](file:///c:/Users/Ashutosh/Downloads/DBMS%20Project/pilot_workload_engine/src/data/cogpilot_loader.py) implemented and ready for authenticated data files. |
| **Cessna 172 Multimodal Dataset** | Wearable EEG, ECG, EDA, ADS-B | **Methodology Paper** | Kinematics profile matched in flight dynamics pipeline. |
| **VR Flight Sim Cognitive Load** | Pupillometry and EEG band powers | **In-lab Academic Study** | Eye dynamics modeled in ocular encoder. |

---

## 2. Machine Learning Models Developed

In accordance with Section 5 of the specification, we implemented and trained five distinct ML modules:

### 2.1 Classical ML Benchmarks (`src/models/classical_baselines.py`)
Tabular feature extractor [`MultiModalFeatureExtractor`](file:///c:/Users/Ashutosh/Downloads/DBMS%20Project/pilot_workload_engine/src/features/feature_extractor.py) extracts **57 features per window**:
- **EEG Spectral Bands:** $\delta, \theta, \alpha, \beta, \gamma$, Theta-to-Beta Ratio (TBR), Alpha-to-Beta Ratio (ABR), Engagement Index.
- **Autonomic HRV & Arousal:** Mean HR, RMSSD, SDNN, respiration rate proxy, GSR mean, slope, and dynamic range.
- **Kinematics:** Vertical $G_z$, turbulence RMS, vertical jerk ($dG/dt$), pitch rate, roll rate, and altitude rate.

Models trained and saved in `review_artifacts/models/`:
- **LightGBM Classifier** (`class_weight="balanced"`)
- **XGBoost Classifier** (`objective="multi:softprob"`)
- **Extra Trees Classifier**
- **Random Forest Classifier** (`class_weight="balanced_subsample"`)

### 2.2 Deep Multi-Modal Attention Classifier (`src/models/workload_classifier.py`)
- Spatial-temporal **EEGNet** (17 channels)
- Cardio **1D ResNet** (ECG, Respiration, GSR)
- Ocular 1D Temporal Convolution
- Flight Context Kinematics MLP
- **Cross-Modal Multi-Head Self-Attention ($H=4$)**
- **Multi-Class Focal Loss ($\gamma = 2.0$)** with automated inverse-frequency class weighting.
- Saved checkpoint: [`checkpoints/best_workload_model.pt`](file:///c:/Users/Ashutosh/Downloads/DBMS%20Project/pilot_workload_engine/checkpoints/best_workload_model.pt).

### 2.3 Kalman Filter Sensor Fusion Engine (`src/engine/sensor_fusion.py`)
- Tracks unobserved true cognitive load via a **1D Linear Kalman Filter**:
  - State prediction: $\hat{x}_{k|k-1} = \hat{x}_{k-1|k-1}$
  - Covariance prediction: $P_{k|k-1} = P_{k-1|k-1} + Q$
  - Multi-sensor composite observation: $z_t = 0.45 s_{\text{eeg}} + 0.30 s_{\text{hrv}} + 0.25 s_{\text{gsr}}$
  - Measurement update: $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - \hat{x}_{k|k-1})$
- Outputs continuous **Cognitive Load Index ($CLI \in [0.0, 1.0]$)** and state tiers:
  - `NORMAL`: $CLI < 0.40$
  - `ELEVATED_LOAD`: $0.40 \le CLI < 0.75$
  - `CRITICAL_OVERLOAD`: $CLI \ge 0.75$
- Emits schema-compliant database records matching the `CognitiveStateSnapshot` relational table.

### 2.4 Flight-Phase Segmentation & Kinematic Clustering (`src/features/flight_phase_segmenter.py`)
- Categorizes flight profiles into: `TAXI_GROUND`, `CLIMB`, `CRUISE`, `DESCENT`, and `EMERGENCY_UPSET`.
- Employs a deterministic aerodynamics state machine + Gaussian Mixture Model (GMM) clustering.
- Generates `flight_phase` tags and `is_emergency_simulated` booleans for SQL `GROUP BY` longitudinal analysis.

### 2.5 Ocular / PERCLOS Computer Vision Model (`src/models/ocular_vision.py`)
- Lightweight CNN (`OcularVigilanceNet`) classifying open vs closed eye crops ($< 1\text{ ms}$ latency).
- Real-time `PERCLOSCalculator` tracking rolling % eye closure over a 60s sliding window, voluntary blinks (100–400ms), and micro-sleep events ($> 1.5\text{s}$).
- Saved checkpoint: [`checkpoints/best_ocular_vigilance_model.pt`](file:///c:/Users/Ashutosh/Downloads/DBMS%20Project/pilot_workload_engine/checkpoints/best_ocular_vigilance_model.pt).

---

## 3. Comparative Benchmark Results

Evaluated on the held-out test split of the real aviation dataset (346 windows, 57 features):

| Model Identifier | Architecture | Balanced Acc (%) | Macro F1 | Log Loss | Latency (p50) | Latency (p99) | Avionics Budget (<50ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | Balanced Gradient Boosted Trees | **81.91%** | **0.8432** | **0.1600** | **0.477 ms** | **0.803 ms** | ✅ PASSED ($> 60\times$ margin) |
| **XGBoost** | Multi-Class Softmax Trees | **76.94%** | **0.8193** | **0.1561** | **0.354 ms** | **2.481 ms** | ✅ PASSED ($> 140\times$ margin) |
| **ExtraTrees** | Extremely Randomized Trees | **78.16%** | **0.7897** | **0.3901** | **15.968 ms** | **32.083 ms** | ✅ PASSED |
| **RandomForest** | Balanced Subsample Forest | **59.96%** | **0.6485** | **0.2241** | **15.923 ms** | **32.661 ms** | ✅ PASSED |
| **DeepMultimodalNet** | EEGNet + Cardio ResNet + MHA Fusion | **74.31%** | **0.5837** | **0.6431** | **1.922 ms** | **3.659 ms** | ✅ PASSED ($> 13\times$ margin) |

### Top Predictive Physiological Biomarkers (MDI Importance)
1. `ecg_rms` (12.56%) — Cardiovascular electrical energy
2. `resp_mean` (11.03%) — Respiration baseline displacement
3. `ecg_mean` (9.39%) — Cardiac potential baseline
4. `gsr_max` (4.52%) — Peak electrodermal skin conductance
5. `gsr_mean` (4.36%) — Sympathetic arousal tonic level
6. `gsr_slope` (4.06%) — Acute stress onset velocity

### LightGBM Confusion Matrix (Held-out Test Windows)
```
                 Predicted: [A]   [B]   [C]   [D]
Actual [A: Base]:            171     0     2     7
Actual [B: CA]  :              1     5     0     0
Actual [C: DA]  :              1     0   141     0
Actual [D: SS]  :              9     0     0     9
----------------------------------------------------------
Key: [A] Baseline, [B] Channelized Attention, [C] Diverted Attention, [D] Startle/Surprise
```

---

## 4. Verification & Testing

The complete test suite expanded from 24 to **39 automated unit tests**, covering:
- Resampling, synchronization, and window boundary guards (`tests/test_synchronization.py`)
- Feature extraction and dataset serialization (`tests/test_dataloader.py`)
- Neural network forward and backward gradient passes (`tests/test_models.py`)
- Multi-class Focal Loss and class weighting (`tests/test_losses_and_training.py`)
- Decision engine 3-tier display decluttering state machine (`tests/test_decision_engine.py`)
- **Kalman filter sensor fusion and covariance convergence** (`tests/test_sensor_fusion.py`)
- **Flight-phase aerodynamics classification and GMM clustering** (`tests/test_flight_phase.py`)
- **Ocular CNN forward pass, PERCLOS calculation, and microsleep alarms** (`tests/test_ocular_vision.py`)

```bash
python -m unittest discover tests -v
# Result: Ran 39 tests in 2.847s -> OK (100% Passing)
```

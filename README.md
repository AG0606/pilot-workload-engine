# Multimodal Pilot Cognitive & Physical Workload Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-39%2F39%20Passing-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/Latency%20(p50)-0.35%20ms-success.svg)]()
[![Avionics Budget](https://img.shields.io/badge/Avionics%20Budget-%3C%2050%20ms-informational.svg)]()
[![Kaggle Pipeline](https://img.shields.io/badge/Kaggle-Completed-blue.svg)](https://www.kaggle.com/code/ag0606/pilot-workload-engine-phase1)

An end-to-end software platform designed to predict real-time pilot cognitive and physical workload from multimodal flight telemetry and bio-signals, feeding a downstream multi-agent decision engine that automates cockpit primary flight display (PFD) decluttering, advisory chimes, and flight task allocation during high-stress flight regimes.

---

## 1. Flight Regimes & Cognitive States

| Label ID | State | Description | Avionics Response |
| :--- | :--- | :--- | :--- |
| **0** | **Baseline (A)** | Nominal flight cruise conditions; balanced cognitive reserve. | `LEVEL_0_FULL` display symbology; pilot in direct manual control. |
| **1** | **Channelized Attention (B / CA)** | Cognitive tunnel vision; pilot fixes on a single gauge or problem. | `LEVEL_1_MODERATE` declutter; triggers `CROSS_CHECK_CHIME`; co-pilot assist. |
| **2** | **Diverted Attention (C / DA)** | Distraction from primary flight path (e.g., secondary system failure). | `LEVEL_1_MODERATE` declutter; triggers `HEADS_UP_CHIME`; heads-up advisory. |
| **3** | **Startle / Surprise (D / SS)** | Acute stress response to unexpected flight regime transition or upset. | `LEVEL_2_ESSENTIALS` declutter ("Basic Six" only); autopilot auto-assist. |

---

## 2. System Architecture

```mermaid
graph TD
    subgraph S1["1. Raw Cockpit Ingestion (Multi-Rate)"]
        EEG["Raw EEG (256 Hz, 17 leads)"]
        ECG["ECG & Autonomic (256 Hz: ECG, Resp, GSR)"]
        EYE["Ocular Dynamics (60 Hz: Pupil, Blinks)"]
        QAR["Flight Kinematics (10 Hz: G-Load, Pitch, Roll)"]
    end

    subgraph S2["2. Synchronization & Windowing"]
        SYNC["MultiModalSynchronizer (20 Hz Reference Grid)"]
        WIN["SlidingWindowExtractor (4.0s Window, 1.0s Stride)"]
        GUARD["Boundary Guard (Rejects Session Transitions & Gaps > 0.15s)"]
    end

    subgraph S3["3. Deep Neural Architecture"]
        EEG_ENC["EEGNetEncoder (Temporal + Spatial Depthwise Conv)"]
        CARDIO_ENC["CardioEncoder (1D Residual Convolutions)"]
        OCULAR_ENC["OcularEncoder (Gaze Temporal Conv)"]
        CONTEXT_ENC["FlightContextEncoder (Kinematic MLP)"]
        ATTN["CrossModalAttentionFusion (Multi-Head Attention H=4)"]
        HEAD["Classification MLP Head"]
    end

    subgraph S4["4. Optimization & Decision Engine"]
        FOCAL["Focal Loss (gamma=2.0 + Inverse Class Weights)"]
        DECISION["CockpitDecisionEngine (Hysteresis Filter)"]
        ACT1["Display Decluttering (Level 0, 1, 2)"]
        ACT2["Auditory Chimes (Cross-Check, Heads-Up)"]
        ACT3["Task Allocation (Pilot vs Autopilot Assist)"]
    end

    EEG --> SYNC
    ECG --> SYNC
    EYE --> SYNC
    QAR --> SYNC
    SYNC --> WIN
    WIN --> GUARD
    GUARD --> EEG_ENC
    GUARD --> CARDIO_ENC
    GUARD --> OCULAR_ENC
    GUARD --> CONTEXT_ENC
    EEG_ENC --> ATTN
    CARDIO_ENC --> ATTN
    OCULAR_ENC --> ATTN
    CONTEXT_ENC --> ATTN
    ATTN --> HEAD
    HEAD --> FOCAL
    HEAD --> DECISION
    DECISION --> ACT1
    DECISION --> ACT2
    DECISION --> ACT3
```

---

## 3. Key Technical Innovations

1. **Multi-Rate Synchronization to 20 Hz:**
   - Real avionics glass displays refresh at 20–60 Hz. Downsampling multi-channel continuous waveforms to a 20 Hz reference time grid retains all cognitive frequency bands up to 10 Hz (Alpha and Theta bands) while reducing window dimensionality from 1,024 to 80 points.
   - Nearest-neighbor interpolation preserves discrete categorical events without artifactual intermediate values.
2. **Boundary & Gap Integrity Guards:**
   - Validates session continuity; windows spanning flight experiment boundaries or gaps $> 0.15\text{ s}$ are rejected to prevent training data leakage.
3. **Multi-Class Focal Loss with Dynamic Weighting:**
   \[
   \mathcal{L}_{\text{focal}} = -\alpha_{y} (1 - p_{t})^\gamma \log(p_{t}), \quad \gamma = 2.0
   \]
   - Directly solves the severe 80% Baseline imbalance by suppressing gradient updates from well-classified nominal samples.
4. **Cross-Modal Attention Fusion:**
   - Dynamic multi-head self-attention ($H=4$) over learnable modality tokens ($D_{\text{fused}}=256$) contextualizes physiological indicators using aircraft kinematic stress (e.g., G-forces vs. straight-and-level flight).
5. **Real-Time Avionics Performance:**
   - Single-window inference latency: **$p_{50} = 2.78\text{ ms}$**, **$p_{99} = 3.54\text{ ms}$** on CPU ($> 14\times$ faster than the $< 50.0\text{ ms}$ avionics budget).

---

## 4. Cockpit Decision Engine & Decluttering Taxonomy

| Predicted State | Trigger Threshold | Display Declutter Level | Active Display Elements | Auditory Alert | Task Allocation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (A)** | $P > 0.60$ | `LEVEL_0_FULL` | All active: PFD, ND, Weather, TCAS, Engines, COM | `NONE` | `PILOT_FLYING` |
| **Channelized Attention (B)** | $P > 0.40$ | `LEVEL_1_MODERATE` | Suppresses secondary gauges; highlights attitude tape | `CROSS_CHECK_CHIME` | `SHARED_COCKPIT` |
| **Diverted Attention (C)** | $P > 0.40$ | `LEVEL_1_MODERATE` | Suppresses weather radar & non-threat TCAS; dims COM | `HEADS_UP_CHIME` | `SHARED_COCKPIT` |
| **Startle / Surprise (D)** | $P > 0.30$ | `LEVEL_2_ESSENTIALS` | Minimalist "Basic Six" attitude symbology only | `ATTITUDE_RECOVERY_WARNING` | `AUTOPILOT_AUTO_ASSIST` |

---

## 5. Repository Layout

```
pilot_workload_engine/
├── config/
│   └── data_config.yaml            # Sampling rates, window sizes, channel mappings
├── src/
│   ├── data/
│   │   ├── base_loader.py          # Abstract ingestion interface
│   │   ├── kaggle_aviation_loader.py # Chunked streaming for 1.23 GB flight logs
│   │   ├── cogpilot_loader.py      # CogPilot multi-modal loader
│   │   ├── telemetry_loader.py     # NASA/QAR flight dynamics loader
│   │   └── synchronizer.py         # MultiModalSynchronizer & SlidingWindowExtractor
│   ├── features/
│   │   ├── spectral_eeg.py         # Welch PSD bands, Theta/Beta & Alpha/Beta ratios
│   │   ├── hrv_features.py         # ECG R-peaks, RMSSD, SDNN, LF/HF balance
│   │   └── flight_dynamics.py      # Turbulence RMS, vertical jerk, angular rates
│   ├── models/
│   │   ├── eeg_encoder.py          # EEGNet spatial-temporal convolutional encoder
│   │   ├── cardio_encoder.py       # 1D Residual block convolutional encoder
│   │   ├── ocular_encoder.py       # Gaze & pupillometry temporal encoder
│   │   ├── context_encoder.py      # Aircraft kinematics context MLP
│   │   ├── fusion_network.py       # CrossModalAttentionFusion (Multi-Head H=4)
│   │   └── workload_classifier.py  # Unified multimodal classifier network
│   ├── training/
│   │   ├── losses.py               # FocalLoss & inverse frequency class weighting
│   │   ├── metrics.py              # Kaggle log loss, balanced accuracy, Macro F1
│   │   └── trainer.py              # Mixed-precision training loop & checkpointing
│   └── engine/
│       └── decision_engine.py      # 3-Tier display decluttering state machine
├── tests/                          # 24/24 Automated unit tests
├── scripts/
│   ├── process_raw_data.py         # CLI preprocessing & window extraction
│   ├── train_model.py              # CLI training & evaluation engine
│   ├── build_notebook.py           # Automated Kaggle notebook compiler
│   └── generate_review_artifacts.py # Review artifact generator
├── review_artifacts/               # Generated confusion matrices & declutter timelines
├── notebooks/                      # Kaggle notebook & metadata
├── REVIEW_1_DOSSIER.md             # Complete Technical Dossier & Review 1 Defense Q&A
└── requirements.txt
```

---

## 6. Quickstart & Usage

### Installation
```bash
git clone https://github.com/AG0606/pilot-workload-engine.git
cd pilot-workload-engine
pip install -r requirements.txt
```

### Running Unit Tests (39/39 Passing)
```bash
python -m unittest discover tests -v
```

### Running the End-to-End Training & Evaluation Pipeline
```bash
# 1. Ingestion & Multi-Rate Synchronization from real flight records
python scripts/process_raw_data.py --raw-csv data/raw/train.csv --crew 1 --output-dir output

# 2. Train Classical Tree Ensemble Benchmarks (LightGBM, XGBoost, ExtraTrees, RandomForest)
python scripts/train_classical_models.py --data-path output/pilot_workload_dataset.pt --output-dir review_artifacts

# 3. Train Deep Multi-Modal Attention Network (EEGNet + Cardio ResNet + Cross-Modal Attention)
python scripts/train_model.py --data-path output/pilot_workload_dataset.pt --epochs 10 --batch-size 32 --output-dir checkpoints

# 4. Train Ocular / PERCLOS Vision Model
python scripts/train_perclos_model.py --output-dir checkpoints --epochs 5

# 5. Run Comprehensive Side-by-Side Model Benchmark & Avionics Latency Profiling
python scripts/evaluate_all_models.py --data-path output/pilot_workload_dataset.pt
```

---

## 7. Benchmarks & Validation Results

| Model Identifier | Architecture Layer | Balanced Acc (%) | Macro F1 | Multi-Class Log Loss | Latency (p50) | Latency (p99) | Avionics Status (<50ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | Balanced Gradient Boosted Trees | **81.91%** | **0.8432** | **0.1600** | **0.477 ms** | **0.803 ms** | ✅ PASSED ($> 60\times$ safety margin) |
| **XGBoost** | Multi-Class Softmax Trees | **76.94%** | **0.8193** | **0.1561** | **0.354 ms** | **2.481 ms** | ✅ PASSED ($> 140\times$ safety margin) |
| **ExtraTrees** | Extremely Randomized Trees | **78.16%** | **0.7897** | **0.3901** | **15.968 ms** | **32.083 ms** | ✅ PASSED |
| **RandomForest** | Balanced Subsample Forest | **59.96%** | **0.6485** | **0.2241** | **15.923 ms** | **32.661 ms** | ✅ PASSED |
| **DeepMultimodalNet** | EEGNet + Cardio ResNet + Attention Fusion | **74.31%** | **0.5837** | **0.6431** | **1.922 ms** | **3.659 ms** | ✅ PASSED ($> 13\times$ safety margin) |

* **Full Walkthrough & Technical Report:** See [`ML_WALKTHROUGH.md`](ML_WALKTHROUGH.md) and [`REVIEW_1_DOSSIER.md`](REVIEW_1_DOSSIER.md).

---

## 8. License
MIT License. Developed for aerospace cognitive state prediction and intelligent cockpit decision support systems.

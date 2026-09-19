# e-Pilot Multi-Modal Workload Prediction: Complete Model Benchmark Report

## 1. Executive Performance Benchmark

| Model Identifier | Architecture Layer | Balanced Acc (%) | Macro F1 | Multi-Class Log Loss | Latency (p50) | Latency (p99) | Avionics Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | Classical Gradient Boosted / Tree Ensemble | **81.91%** | 0.8432 | 0.1600 | `0.477 ms` | `0.803 ms` | ✅ PASSED |
| **XGBoost** | Classical Gradient Boosted / Tree Ensemble | **76.94%** | 0.8193 | 0.1561 | `0.354 ms` | `2.481 ms` | ✅ PASSED |
| **ExtraTrees** | Classical Gradient Boosted / Tree Ensemble | **78.16%** | 0.7897 | 0.3901 | `15.968 ms` | `32.083 ms` | ✅ PASSED |
| **RandomForest** | Classical Gradient Boosted / Tree Ensemble | **59.96%** | 0.6485 | 0.2241 | `15.923 ms` | `32.661 ms` | ✅ PASSED |
| **DeepMultimodalAttentionNet** | Spatial-Temporal EEGNet + Cardio ResNet + Cross-Modal Attention | **74.31%** | 0.5837 | 0.6431 | `1.922 ms` | `3.659 ms` | ✅ PASSED |

---

## 2. Best Model Confusion Matrix

```
CONFUSION MATRIX (LightGBM - Held-out Aviation Test Set):
----------------------------------------------------------
                 Predicted: [A]   [B]   [C]   [D]
Actual [A: Base]:            171     0     2     7
Actual [B: CA]  :              1     5     0     0
Actual [C: DA]  :              1     0   141     0
Actual [D: SS]  :              9     0     0     9
----------------------------------------------------------
Key: [A] Baseline, [B] Channelized Attention, [C] Diverted Attention, [D] Startle/Surprise

```

---

## 3. Real-Time Avionics Constraint Verification
* **Hard Real-Time Constraint:** $< 50.0\text{ ms}$ (Glass cockpit PFD render loop)
* **Fastest Model (XGBoost):** $p_{50} = 0.354\text{ ms}$ ($> 100\times$ safety margin)
* **Highest Accuracy (LightGBM):** $p_{50} = 0.477\text{ ms}$ with **81.91% Balanced Accuracy**

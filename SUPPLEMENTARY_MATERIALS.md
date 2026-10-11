# Supplementary Materials & Technical Appendix

**Manuscript Title:** *e-Pilot: A High-Throughput Database-Centric and Deep Multimodal Architecture for Real-Time Pilot Cognitive Workload Assessment and Cockpit Display Decluttering*

---

## Appendix A: Mathematical Formulation of BCNF Normalization

### 1. Functional Dependency Set across Entities
Let the global relational universe $\mathcal{U}$ be governed by the minimal cover of functional dependencies $\mathcal{F}_{\min}$:

1. **`aircraft`**:
   $$\text{tail\_number} \to \{\text{model}, \text{manufacturer}, \text{max\_gross\_weight\_kg}, \text{seating\_capacity}, \text{certification\_status}, \text{created\_at}\}$$
   *Candidate Key:* $\{\text{tail\_number}\}$. Determinant is a superkey $\implies$ **BCNF**.

2. **`pilot`**:
   $$\text{pilot\_id} \to \{\text{license\_number}, \text{first\_name}, \text{last\_name}, \text{total\_flight\_hours}, \text{medical\_cert\_expiry}, \text{created\_at}\}$$
   $$\text{license\_number} \to \{\text{pilot\_id}, \text{first\_name}, \text{last\_name}, \dots\}$$
   *Candidate Keys:* $\{\text{pilot\_id}\}$, $\{\text{license\_number}\}$. Both determinants are superkeys $\implies$ **BCNF**.

3. **`flight`**:
   $$\text{flight\_id} \to \{\text{flight\_number}, \text{aircraft\_tail\_number}, \text{origin\_icao}, \text{destination\_icao}, \text{scheduled\_departure}, \dots\}$$
   *Candidate Key:* $\{\text{flight\_id}\}$. Determinant is a superkey $\implies$ **BCNF**.

4. **`crew_assignment`**:
   $$\text{assignment\_id} \to \{\text{flight\_id}, \text{pilot\_id}, \text{crew\_role}, \text{seat\_position}\}$$
   $$\{\text{flight\_id}, \text{seat\_position}\} \to \{\text{assignment\_id}, \text{pilot\_id}, \text{crew\_role}\}$$
   $$\{\text{flight\_id}, \text{pilot\_id}\} \to \{\text{assignment\_id}, \text{crew\_role}, \text{seat\_position}\}$$
   *Candidate Keys:* $\{\text{assignment\_id}\}$, $\{\text{flight\_id}, \text{seat\_position}\}$, $\{\text{flight\_id}, \text{pilot\_id}\}$. All determinants are superkeys $\implies$ **BCNF**.

5. **`sensor_device`**:
   $$\text{device\_id} \to \{\text{device\_serial}, \text{device\_type}, \text{hardware\_model}, \text{firmware\_version}, \text{sampling\_rate\_hz}, \dots\}$$
   $$\text{device\_serial} \to \{\text{device\_id}, \text{device\_type}, \dots\}$$
   *Candidate Keys:* $\{\text{device\_id}\}$, $\{\text{device\_serial}\}$. Determinants are superkeys $\implies$ **BCNF**.

6. **`sensor_reading`**:
   $$\text{reading\_id} \to \{\text{flight\_id}, \text{device\_id}, \text{pilot\_id}, \text{timestamp}, \text{channel\_name}, \text{raw\_value}, \text{unit}, \text{quality\_index}\}$$
   *Candidate Key:* $\{\text{reading\_id}\}$. Determinant is a superkey $\implies$ **BCNF**.

7. **`cognitive_state_snapshot`**:
   $$\text{snapshot\_id} \to \{\text{flight\_id}, \text{pilot\_id}, \text{timestamp}, \text{predicted\_state}, \text{confidence\_score}, \text{cognitive\_load\_index}, \dots\}$$
   *Candidate Key:* $\{\text{snapshot\_id}\}$. Determinant is a superkey $\implies$ **BCNF**.

8. **`alert`**:
   $$\text{alert\_id} \to \{\text{snapshot\_id}, \text{flight\_id}, \text{pilot\_id}, \text{timestamp}, \text{severity\_level}, \text{declutter\_level}, \dots\}$$
   $$\text{snapshot\_id} \to \{\text{alert\_id}, \text{flight\_id}, \text{pilot\_id}, \dots\}$$
   *Candidate Keys:* $\{\text{alert\_id}\}$, $\{\text{snapshot\_id}\}$. Determinants are superkeys $\implies$ **BCNF**.

9. **`alert_audit_log`**:
   $$\text{audit\_id} \to \{\text{alert\_id}, \text{action\_taken}, \text{operator\_notes}, \text{logged\_timestamp}\}$$
   *Candidate Key:* $\{\text{audit\_id}\}$. Determinant is a superkey $\implies$ **BCNF**.

---

## Appendix B: Deep Multimodal Classifier Hyperparameters

| Sub-Module / Block | Layer / Operation | Parameters / Hyperparameters | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input Window** | Synchronized Batch | 4.0s @ 20 Hz = 80 Samples | $[B, C, 80]$ |
| **EEGNet Encoder** | 1D Temporal Convolution | Kernel: $1 \times 15$, Filters: 8, Stride: 1 | $[B, 8, 17, 80]$ |
| | BatchNorm2d + Spatial Depthwise | Kernel: $17 \times 1$, Depth Multiplier: 2 | $[B, 16, 1, 80]$ |
| | ELU + AvgPool2d | Pool: $1 \times 4$, Stride: $1 \times 4$ | $[B, 16, 1, 20]$ |
| | Separable Conv + Dropout | Kernel: $1 \times 16$, Dropout: $p=0.25$ | $[B, 16, 1, 20]$ |
| | Linear Projection | Linear $(320 \to 128)$ + LayerNorm | $[B, 128]$ |
| **Cardio ResNet** | 1D Convolution | Kernel: 7, Filters: 32, Stride: 1 | $[B, 32, 80]$ |
| | Residual Block 1 | 2x Conv1D(32, 32, k=3) + Skip + LeakyReLU | $[B, 32, 80]$ |
| | MaxPool1D + Residual Block 2 | Pool: 2, 2x Conv1D(32, 64, k=3) + Skip | $[B, 64, 40]$ |
| | AdaptiveAvgPool1D + Linear | AdaptiveAvgPool(1) + Linear $(64 \to 128)$ | $[B, 128]$ |
| **Ocular Encoder** | 1D Temporal Convolution | 2x Conv1D(4, 32, k=5) + MaxPool(2) | $[B, 32, 40]$ |
| | Conv1D + AdaptiveAvgPool1D | Conv1D(32, 64, k=3) + AdaptiveAvgPool(1) | $[B, 64]$ |
| | Linear Projection | Linear $(64 \to 128)$ + LeakyReLU | $[B, 128]$ |
| **Flight Context** | Kinematics Projection MLP | Linear $(8 \times 80 \to 128)$ + BatchNorm + LeakyReLU | $[B, 128]$ |
| **Cross-Modal MHA** | Modality Token Stacking | 4 Modality Tokens + Positional Embeddings | $[B, 4, 128]$ |
| | Multi-Head Self-Attention | Heads $H=4$, Head Dim $d_k=32$, Dropout $0.1$ | $[B, 4, 128]$ |
| | Feed-Forward Layer | Linear $(128 \to 256) \to$ LeakyReLU $\to$ Linear $(256 \to 128)$ | $[B, 4, 128]$ |
| | Pooling & Output MLP | Flatten $(4 \times 128 = 512 \to 256)$ | $[B, 256]$ |
| **Classifier Head** | Multi-Layer Perceptron | Linear $(256 \to 64) \to$ BatchNorm $\to$ Linear $(64 \to 4)$ | $[B, 4]$ |

---

## Appendix C: Per-Class Precision, Recall, and F1 Scores

### 1. LightGBM Classification Report (Held-Out Test Set)
- **Baseline (Class A):** Precision = $0.940$, Recall = $0.950$, F1 = $0.945$ (Support = 180)
- **Channelized Attention (Class B):** Precision = $1.000$, Recall = $0.833$, F1 = $0.909$ (Support = 6)
- **Diverted Attention (Class C):** Precision = $0.986$, Recall = $0.993$, F1 = $0.990$ (Support = 142)
- **Startle / Surprise (Class D):** Precision = $0.563$, Recall = $0.500$, F1 = $0.529$ (Support = 18)
- **Overall Macro Average:** Precision = $0.872$, Recall = $0.819$, Macro F1 = **$0.843$**

### 2. XGBoost Classification Report
- **Baseline (Class A):** Precision = $0.935$, Recall = $0.967$, F1 = $0.951$ (Support = 180)
- **Channelized Attention (Class B):** Precision = $1.000$, Recall = $0.667$, F1 = $0.800$ (Support = 6)
- **Diverted Attention (Class C):** Precision = $0.986$, Recall = $1.000$, F1 = $0.993$ (Support = 142)
- **Startle / Surprise (Class D):** Precision = $0.667$, Recall = $0.444$, F1 = $0.533$ (Support = 18)
- **Overall Macro Average:** Precision = $0.897$, Recall = $0.769$, Macro F1 = **$0.819$**

---

## Appendix D: Step-by-Step Reproduction Guide

To independently reproduce the entire test suite and empirical database benchmarks:

```bash
# 1. Clone repository and navigate to workspace
cd pilot_workload_engine

# 2. Run the complete automated test suite (47 unit tests)
python -m unittest discover tests -v
# Verified Result: Ran 47 tests in ~5.7s -> 100% OK

# 3. Execute the database benchmarking suite
python -m src.database.benchmark_db
# Generates review_artifacts/db_benchmark_results.json
```

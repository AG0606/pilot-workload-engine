import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure IEEE-style publication fonts and styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrices():
    """Generates Figure 1: Test Set Confusion Matrix."""
    # Data from held-out test evaluation (346 windows)
    # Classes: [A] Baseline, [B] Channelized, [C] Diverted, [D] Startle
    cm_lgbm = np.array([
        [171, 0, 2, 7],
        [1, 5, 0, 0],
        [1, 0, 141, 0],
        [9, 0, 0, 9],
    ])
    
    classes = ["Baseline (A)", "Channelized (B)", "Diverted (C)", "Startle (D)"]
    cm_norm = cm_lgbm.astype("float") / cm_lgbm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Raw counts
    sns.heatmap(cm_lgbm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title("(a) Raw Counts (Test Set $N=346$)")
    ax1.set_ylabel("Ground Truth Flight Regime")
    ax1.set_xlabel("Predicted Regime")

    # Normalized recall
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title("(b) Normalized True Positive Rate (Recall)")
    ax2.set_ylabel("Ground Truth Flight Regime")
    ax2.set_xlabel("Predicted Regime")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_confusion_matrices.pdf")
    plt.savefig(FIGURES_DIR / "fig1_confusion_matrices.png")
    plt.close()
    print("Generated: figures/fig1_confusion_matrices.pdf & png")


def plot_biomarker_importance():
    """Generates Figure 2: Top Physiological Biomarkers Feature Importance."""
    biomarkers = [
        ("ECG Energy (RMS)", 12.56, "Cardiovascular"),
        ("Respiration Baseline Shift", 11.03, "Respiratory"),
        ("Cardiac Polarization (ECG Mean)", 9.39, "Cardiovascular"),
        ("Peak Electrodermal Conductance (GSR)", 4.52, "Autonomic GSR"),
        ("Tonic Skin Conductance (GSR Mean)", 4.36, "Autonomic GSR"),
        ("Acute Stress Velocity (GSR Slope)", 4.06, "Autonomic GSR"),
        ("Theta/Beta Ratio (TBR Cz)", 3.84, "Neurological EEG"),
        ("Alpha/Beta Ratio (ABR Fz)", 3.42, "Neurological EEG"),
        ("Engagement Index (Beta/(Alpha+Theta))", 3.15, "Neurological EEG"),
        ("Vertical Jerk (dG/dt)", 2.94, "Flight Kinematics"),
        ("Turbulence Intensity (RMS Gz)", 2.68, "Flight Kinematics"),
        ("Heart Rate Variability (RMSSD)", 2.51, "Cardiovascular"),
    ]

    names = [b[0] for b in reversed(biomarkers)]
    values = [b[1] for b in reversed(biomarkers)]
    categories = [b[2] for b in reversed(biomarkers)]

    color_map = {
        "Cardiovascular": "#D9534F",
        "Respiratory": "#5BC0DE",
        "Autonomic GSR": "#F0AD4E",
        "Neurological EEG": "#337AB7",
        "Flight Kinematics": "#5CB85C",
    }
    bar_colors = [color_map[c] for c in categories]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(names, values, color=bar_colors, edgecolor="black", linewidth=0.6, height=0.7)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%",
                va="center", ha="left", fontsize=8.5, color="#333333")

    ax.set_xlabel("Mean Decrease in Impurity (MDI Importance %)")
    ax.set_title("Top Physiological & Kinematic Biomarkers for Pilot Cognitive Workload")
    ax.set_xlim(0, 15)

    # Custom legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[cat], edgecolor="black", label=cat)
        for cat in ["Neurological EEG", "Cardiovascular", "Autonomic GSR", "Respiratory", "Flight Kinematics"]
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_biomarker_importance.pdf")
    plt.savefig(FIGURES_DIR / "fig2_biomarker_importance.png")
    plt.close()
    print("Generated: figures/fig2_biomarker_importance.pdf & png")


def plot_latency_comparison():
    """Generates Figure 3: Real-Time Avionics Latency Benchmark vs 50ms Budget."""
    models = ["XGBoost", "LightGBM", "Deep Multimodal Net\n(MHA Fusion)", "ExtraTrees", "Random Forest"]
    p50_latencies = [0.354, 0.477, 1.922, 15.968, 15.923]
    p99_latencies = [2.481, 0.803, 3.659, 32.083, 32.661]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(x - width/2, p50_latencies, width, label="Median Latency ($p_{50}$)", color="#337AB7", edgecolor="black")
    rects2 = ax.bar(x + width/2, p99_latencies, width, label="Tail Latency ($p_{99}$)", color="#D9534F", edgecolor="black")

    # Avionics budget line
    ax.axhline(50.0, color="#C9302C", linestyle="--", linewidth=1.8, label="Avionics Real-Time Budget ($< 50.0$ ms)")

    ax.set_ylabel("Inference Latency (Milliseconds)")
    ax.set_title("Single-Window Inference Latency vs. Avionics Glass Cockpit Budget")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100.0)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    # Annotate margin
    ax.text(0, 0.55, "141x Safety Margin", ha="center", fontsize=8, color="#1B4F72", fontweight="bold")
    ax.text(1, 0.70, "104x Safety Margin", ha="center", fontsize=8, color="#1B4F72", fontweight="bold")
    ax.text(2, 2.5, "26x Safety Margin", ha="center", fontsize=8, color="#1B4F72", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_latency_comparison.pdf")
    plt.savefig(FIGURES_DIR / "fig3_latency_comparison.png")
    plt.close()
    print("Generated: figures/fig3_latency_comparison.pdf & png")


def plot_cognitive_workload_timeline():
    """Generates Figure 4: Cognitive Load Index (CLI) & 3-Tier Decluttering Dynamics."""
    np.random.seed(42)
    t = np.linspace(0, 120, 120)  # 120 seconds

    # Generate synthetic operational transition: Nominal Cruise -> Turbulence -> Acute Emergency -> Recovery
    raw_stress = np.zeros_like(t)
    for i, sec in enumerate(t):
        if sec < 35:
            raw_stress[i] = 0.18 + 0.08 * np.sin(sec * 0.2) + np.random.randn() * 0.05
        elif sec < 65:
            # Channelized Attention / Distraction
            raw_stress[i] = 0.52 + 0.10 * np.sin(sec * 0.3) + np.random.randn() * 0.08
        elif sec < 90:
            # Acute Startle / Emergency Upset
            raw_stress[i] = 0.88 + 0.06 * np.sin(sec * 0.4) + np.random.randn() * 0.06
        else:
            # Recovery
            raw_stress[i] = 0.30 + 0.05 * np.cos(sec * 0.2) + np.random.randn() * 0.05

    raw_stress = np.clip(raw_stress, 0.0, 1.0)

    # Kalman Filter trajectory
    cli_filtered = np.zeros_like(raw_stress)
    x_hat = 0.20
    p_cov = 0.10
    q = 1e-3
    r = 0.04
    for i, z in enumerate(raw_stress):
        p_cov += q
        k_gain = p_cov / (p_cov + r)
        x_hat += k_gain * (z - x_hat)
        p_cov = (1 - k_gain) * p_cov
        cli_filtered[i] = x_hat

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Axis 1: CLI and Thresholds
    ax1.plot(t, raw_stress, color="#B0BEC5", alpha=0.6, label="Raw Multi-Sensor Observation $z_t$", linewidth=1.0)
    ax1.plot(t, cli_filtered, color="#1565C0", label="Kalman Filtered Cognitive Load Index (CLI)", linewidth=2.0)

    ax1.axhline(0.40, color="#FB8C00", linestyle="--", linewidth=1.2, label="Level 1 Declutter Threshold (CLI = 0.40)")
    ax1.axhline(0.75, color="#D32F2F", linestyle="--", linewidth=1.2, label="Level 2 Essentials Threshold (CLI = 0.75)")

    ax1.set_ylabel("Cognitive Load Index (CLI)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Real-Time Cognitive State Tracking & Cockpit Symbology Actuation")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Axis 2: 3-Tier Display Decluttering State
    declutter_state = np.zeros_like(t)
    for i, cli in enumerate(cli_filtered):
        if cli >= 0.75:
            declutter_state[i] = 2  # Level 2 Essentials
        elif cli >= 0.40:
            declutter_state[i] = 1  # Level 1 Moderate
        else:
            declutter_state[i] = 0  # Level 0 Full

    ax2.step(t, declutter_state, where="post", color="#2E7D32", linewidth=2.0, label="Active Display State")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Level 0\n(Full)", "Level 1\n(Moderate)", "Level 2\n(Essentials)"])
    ax2.set_ylabel("PFD Declutter")
    ax2.set_xlabel("Flight Elapsed Time (Seconds)")
    ax2.set_ylim(-0.3, 2.3)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Annotations
    ax1.annotate("Nominal Cruise", xy=(15, 0.22), xytext=(15, 0.35),
                 arrowprops=dict(arrowstyle="->", color="#333"), fontsize=8.5, ha="center")
    ax1.annotate("Channelized Attention", xy=(50, 0.55), xytext=(50, 0.70),
                 arrowprops=dict(arrowstyle="->", color="#333"), fontsize=8.5, ha="center")
    ax1.annotate("Acute Emergency / Startle", xy=(75, 0.90), xytext=(75, 0.98),
                 arrowprops=dict(arrowstyle="->", color="#D32F2F"), fontsize=8.5, ha="center", color="#D32F2F", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_cognitive_workload_timeline.pdf")
    plt.savefig(FIGURES_DIR / "fig4_cognitive_workload_timeline.png")
    plt.close()
    print("Generated: figures/fig4_cognitive_workload_timeline.pdf & png")


def plot_database_benchmarks():
    """Generates Figure 5: Ingestion Throughput & ACID Transaction Latency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Ingestion throughput bar
    categories = ["Batch Ingestion\n(Single Thread)", "Concurrent Workers\n(4 Parallel Threads)"]
    throughput = [176972.61, 106100.12]

    bars = ax1.bar(categories, throughput, color=["#1976D2", "#388E3C"], edgecolor="black", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 3000, f"{h:,.0f} ops/s",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Throughput (Records Persisted / Second)")
    ax1.set_title("(a) Multi-Stream IoT Ingestion Throughput")
    ax1.set_ylim(0, 210000)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # ACID Transaction Commit Latency Box / Distribution
    percentiles = ["$p_{50}$ (Median)", "$p_{95}$", "$p_{99}$ (Tail)"]
    latencies_ms = [0.1174, 0.2437, 0.5712]

    bars2 = ax2.bar(percentiles, latencies_ms, color=["#0288D1", "#F57C00", "#D32F2F"], edgecolor="black", width=0.55)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.4f} ms",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Commit Latency (Milliseconds)")
    ax2.set_title("(b) Atomic ACID Alert Dispatch Latency")
    ax2.set_ylim(0, 0.75)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_database_benchmarks.pdf")
    plt.savefig(FIGURES_DIR / "fig5_database_benchmarks.png")
    plt.close()
    print("Generated: figures/fig5_database_benchmarks.pdf & png")


if __name__ == "__main__":
    plot_confusion_matrices()
    plot_biomarker_importance()
    plot_latency_comparison()
    plot_cognitive_workload_timeline()
    plot_database_benchmarks()
    print("\nAll publication-quality journal figures generated successfully in figures/")

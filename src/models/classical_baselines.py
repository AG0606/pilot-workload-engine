from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
import xgboost as xgb
import lightgbm as lgb


class ClassicalWorkloadBaselines:
    """Suite of classical machine learning benchmark models for cognitive state classification.

    Directly implements Section 5 of the e-Pilot DBMS architecture specification:
    - Random Forest (with class balancing)
    - Extreme Gradient Boosting (XGBoost)
    - Light Gradient Boosted Machine (LightGBM)
    - Extra Trees Classifier
    """

    def __init__(
        self,
        random_state: int = 42,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.random_state = random_state
        self.class_names = class_names or ["Baseline", "Channelized_Attention", "Diverted_Attention", "Startle"]
        self.models: Dict[str, Any] = {}
        self._init_models()

    def _init_models(self) -> None:
        """Initializes balanced tree ensemble models."""
        self.models["RandomForest"] = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            subsample=0.85,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1,
        )

        self.models["ExtraTrees"] = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train_model(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """Trains the specified benchmark model on the tabular feature matrix."""
        if name not in self.models:
            raise KeyError(f"Model {name} not found in available baselines: {list(self.models.keys())}")

        model = self.models[name]
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = (time.perf_counter() - start_time) * 1000.0
        return model, fit_time

    def evaluate_model(
        self,
        name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluates trained model metrics including balanced accuracy, Macro F1, log loss, and latency."""
        if name not in self.models:
            raise KeyError(f"Model {name} not found.")

        model = self.models[name]

        # Profile single-window inference latency across 100 iterations
        latencies_ms: List[float] = []
        single_sample = X_test[0:1]
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model.predict_proba(single_sample)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        p50_latency = float(np.percentile(latencies_ms, 50))
        p99_latency = float(np.percentile(latencies_ms, 99))

        # Overall predictions
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)

        balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # Clip probabilities for stable multi-class log loss
        eps = 1e-15
        y_probs_clipped = np.clip(y_probs, eps, 1.0 - eps)
        # Re-normalize rows to sum to 1
        y_probs_clipped = y_probs_clipped / y_probs_clipped.sum(axis=1, keepdims=True)

        try:
            m_log_loss = float(log_loss(y_test, y_probs_clipped, labels=list(range(len(self.class_names)))))
        except Exception:
            m_log_loss = float("nan")

        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)

        return {
            "model_name": name,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "log_loss": m_log_loss,
            "p50_latency_ms": p50_latency,
            "p99_latency_ms": p99_latency,
            "report": report,
            "y_pred": y_pred,
            "y_probs": y_probs,
        }

    def get_feature_importances(
        self,
        name: str,
        feature_names: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Extracts top predictive physiological and kinematic features."""
        if name not in self.models:
            raise KeyError(f"Model {name} not found.")

        model = self.models[name]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        return []

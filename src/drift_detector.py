import numpy as np
from scipy import stats
from typing import List, Dict, Any

class DataDriftDetector:
    """موتور پایش آماری رانش داده‌ها بر اساس آزمون کولموگوروف-اسمیرنوف"""
    def __init__(self, p_value_threshold: float = 0.05):
        self.threshold = p_value_threshold
        # توزیع مرجع (Baseline) حاصل از مرحله آموزش
        np.random.seed(42)
        self.baseline_data = np.random.normal(0.0, 1.0, size=(500, 4))

    def evaluate_drift(self, incoming_batch: List[List[float]]) -> Dict[str, Any]:
        batch_array = np.array(incoming_batch)
        if batch_array.ndim != 2 or batch_array.shape[1] != 4:
            raise ValueError("Input batch must have shape (N, 4)")

        feature_reports = {}
        drift_count = 0

        for col_idx in range(4):
            ref_col = self.baseline_data[:, col_idx]
            prod_col = batch_array[:, col_idx]
            
            # آزمون آماری دو نمونه‌ای KS
            stat, p_val = stats.ks_2samp(ref_col, prod_col)
            has_drifted = bool(p_val < self.threshold)
            if has_drifted:
                drift_count += 1

            feature_reports[f"feature_{col_idx+1}"] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
                "drift_detected": has_drifted
            }

        return {
            "overall_drift_detected": bool(drift_count > 0),
            "drifted_features_count": drift_count,
            "feature_metrics": feature_reports
        }

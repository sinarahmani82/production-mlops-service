import numpy as np
from tabulate import tabulate
from src.predictor import ModelService
from src.drift_detector import DataDriftDetector

def main():
    print("=== Production MLOps Microservice: Execution Demo ===\n")
    
    # ۱. تست سرویس پیش‌بینی
    predictor = ModelService()
    sample_features = [0.85, -0.42, 1.20, 0.15]
    pred, prob = predictor.predict(sample_features)
    print(f"Sample Input: {sample_features}")
    print(f"Prediction: {pred} | Probability: {prob:.4f}\n")

    # ۲. شبیه‌سازی ترافیک سالم (In-Distribution)
    print("--- Phase 1: Evaluating Healthy Production Batch (No Drift) ---")
    detector = DataDriftDetector()
    healthy_batch = np.random.normal(0.0, 1.0, size=(100, 4)).tolist()
    report_healthy = detector.evaluate_drift(healthy_batch)
    print(f"Overall Drift Detected: {report_healthy['overall_drift_detected']} (Drifted Features: {report_healthy['drifted_features_count']})\n")

    # ۳. شبیه‌سازی ترافیک دچار رانش (Covariate Shift)
    print("--- Phase 2: Evaluating Drifted Production Batch (Covariate Shift Injected) ---")
    drifted_batch = np.random.normal(1.8, 1.5, size=(100, 4)).tolist()
    report_drifted = detector.evaluate_drift(drifted_batch)
    
    table_data = []
    for feat, metrics in report_drifted["feature_metrics"].items():
        status = "DRIFT DETECTED" if metrics["drift_detected"] else "IN-CONTROL"
        table_data.append([feat, metrics["ks_statistic"], metrics["p_value"], status])
        
    print(tabulate(table_data, headers=["Feature", "KS-Statistic", "p-value", "Audit Status"], tablefmt="grid"))
    print(f"\n[Alert]: Overall Drift Triggered: {report_drifted['overall_drift_detected']}")

if __name__ == "__main__":
    main()

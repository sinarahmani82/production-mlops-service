import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    """تست پاسخ‌دهی اندپوینت سلامت سرویس"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_payload():
    """تست صحت پیش‌بینی با ورودی معتبر ۴ فیچری"""
    payload = {"features": [1.2, -0.5, 0.3, 2.1]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0

def test_predict_invalid_payload_validation():
    """تست اعتبارسنجی Pydantic: ارسال ۳ فیچر به جای ۴ فیچر باید خطای ۴۲۲ بدهد"""
    invalid_payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_drift_monitoring_detection():
    """تست آزمون آماری رانش داده‌ها روی ترافیک منحرف‌شده"""
    np.random.seed(42)
    # ساخت یک بسته با توزیع به شدت متفاوت از مرجع
    drifted_batch = np.random.normal(3.5, 1.0, size=(50, 4)).tolist()
    
    response = client.post("/monitor/drift", json={"batch_data": drifted_batch})
    assert response.status_code == 200
    data = response.json()
    assert data["overall_drift_detected"] is True

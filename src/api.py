from fastapi import FastAPI, HTTPException
from src.schemas import InferencePayload, PredictionResponse, BatchPayload, DriftReportResponse
from src.predictor import ModelService
from src.drift_detector import DataDriftDetector

app = FastAPI(
    title="Production MLOps Service",
    description="Clean Architecture ML Inference & Real-time Data Drift Monitoring API",
    version="1.0.0"
)

# نمونه‌سازی از سرویس‌ها
model_service = ModelService()
drift_service = DataDriftDetector(p_value_threshold=0.05)

@app.get("/health")
async def health_check():
    """بررسی وضعیت سلامت کانتینر و سرویس"""
    return {"status": "healthy", "service": "MLOps Production API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sample(payload: InferencePayload):
    """اندپوینت پیش‌بینی بیدرنگ با اعتبارسنجی خودکار"""
    pred, prob = model_service.predict(payload.features)
    return {
        "prediction": pred,
        "probability": round(prob, 4),
        "status": "success"
    }

@app.post("/monitor/drift", response_model=DriftReportResponse)
async def check_drift(payload: BatchPayload):
    """پایش زنده دسته‌ای از ترافیک تولید برای کشف رانش داده‌ها"""
    if len(payload.batch_data) < 10:
        raise HTTPException(status_code=400, detail="Batch size must be at least 10 samples for statistical testing.")
    
    report = drift_service.evaluate_drift(payload.batch_data)
    return {
        "overall_drift_detected": report["overall_drift_detected"],
        "drift_details": report
    }

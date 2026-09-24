from pydantic import BaseModel, Field
from typing import List, Dict, Any

class InferencePayload(BaseModel):
    """اسکیمای اعتبارسنجی ورودی تک‌نمونه برای پیش‌بینی"""
    features: List[float] = Field(..., min_length=4, max_length=4, description="Vector of 4 numerical features")

class PredictionResponse(BaseModel):
    """اسکیمای خروجی پاسخ مدل"""
    prediction: int
    probability: float
    status: str

class BatchPayload(BaseModel):
    """اسکیمای ورودی دسته‌ای برای پایش رانش داده‌ها"""
    batch_data: List[List[float]]

class DriftReportResponse(BaseModel):
    """اسکیمای گزارش سلامت و رانش داده‌ها"""
    overall_drift_detected: bool
    drift_details: Dict[str, Any]

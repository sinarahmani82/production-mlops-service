import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Tuple

class ModelService:
    """سرویس ایزوله‌شده پیش‌بینی مدل بر اساس معماری پاک"""
    def __init__(self):
        # ساخت یک مدل پایه کالیبره‌شده با ۴ فیچر
        np.random.seed(42)
        X_train = np.random.normal(0, 1, size=(200, 4))
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

    def predict(self, features: list) -> Tuple[int, float]:
        X = np.array(features).reshape(1, -1)
        prob = float(self.model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        return pred, prob

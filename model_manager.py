"""
scikit-learn 기반 qPCR Ct값 예측 회귀 모델 관리 모듈.

샘플 수에 따라 적합한 모델을 자동 선택:
  < 10개  → LinearRegression (해석 가능, 과적합 방지)
  10~29개 → GradientBoostingRegressor (비선형, 중소 데이터)
  30개 이상 → RandomForestRegressor (앙상블, 충분한 데이터)

특징 벡터 (4차원):
  [band_intensity, band_area, relative_intensity, band_width]
"""

import json
import os
from datetime import datetime, timezone
from typing import List

import joblib
import numpy as np

from logger import get_logger

log = get_logger("model_manager")
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model/ct_predictor.pkl"
META_PATH = "model/ct_predictor_meta.json"

FEATURE_KEYS = ["band_intensity", "band_area", "relative_intensity", "band_width", "band_height"]


class ModelManager:
    """Ct값 예측 모델의 학습·저장·로드·예측을 담당합니다."""

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.meta: dict = {}
        self._load_if_exists()

    # ── 공개 메서드 ──────────────────────────────────────────────────

    def train(self, features: List[dict], ct_values: List[float]) -> dict:
        """
        훈련 데이터로 회귀 모델을 학습하고 디스크에 저장합니다.

        Args:
            features:   각 이미지의 특징 dict 리스트
            ct_values:  대응하는 실측 Ct값 리스트

        Returns:
            학습 메트릭 dict (r2, rmse, sample_count 등)
        """
        X = self._to_matrix(features)
        y = np.array(ct_values, dtype=float)
        n = len(y)

        base_model = self._select_model(n)
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", base_model)])

        # 교차 검증
        cv = LeaveOneOut() if n <= 10 else min(5, n)
        cv_r2 = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

        # 전체 데이터로 최종 학습
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        self.pipeline = pipeline
        # NaN이 포함된 경우(소규모 LOO CV에서 발생) 0으로 대체
        cv_r2_clean = np.nan_to_num(cv_r2, nan=0.0)

        self.meta = {
            "sample_count": n,
            "model_type": type(base_model).__name__,
            "train_r2": round(float(r2_score(y, y_pred)), 4),
            "train_rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
            "cv_r2_mean": round(float(np.mean(cv_r2_clean)), 4),
            "cv_r2_std": round(float(np.std(cv_r2_clean)), 4),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(FEATURE_KEYS),
        }

        os.makedirs("model", exist_ok=True)
        joblib.dump(self.pipeline, MODEL_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

        log.info("모델 학습 완료 - %s | samples=%d, R²=%.4f, RMSE=%.4f, CV_R²=%.4f±%.4f",
                 self.meta["model_type"], n,
                 self.meta["train_r2"], self.meta["train_rmse"],
                 self.meta["cv_r2_mean"], self.meta["cv_r2_std"])
        return self.meta

    def predict(self, features: dict) -> dict:
        """
        단일 특징 dict로 Ct값을 예측합니다.

        Returns:
            dict: predicted_ct, model_r2, model_rmse
        """
        if self.pipeline is None:
            raise ValueError("학습된 모델이 없습니다. /train 을 먼저 실행하세요.")

        X = self._to_matrix([features])
        # 저장된 모델의 피처 수와 다를 경우 슬라이싱 (하위 호환)
        trained_feature_count = self.meta.get("feature_count", 4)
        if X.shape[1] != trained_feature_count:
            X = X[:, :trained_feature_count]
        predicted_ct = float(self.pipeline.predict(X)[0])

        CT_MIN, CT_MAX = 5.0, 50.0
        if predicted_ct < CT_MIN or predicted_ct > CT_MAX:
            log.warning("Ct값 범위 이탈 (%.2f) → 미검출 처리", predicted_ct)
            raise ValueError(f"예측값({predicted_ct:.2f})이 유효 범위({CT_MIN}–{CT_MAX})를 벗어났습니다.")

        log.info("Ct값 예측 완료 - predicted_ct=%.2f (model=%s)", predicted_ct, self.meta.get("model_type"))
        return {
            "predicted_ct": round(predicted_ct, 2),
            "model_r2": self.meta.get("cv_r2_mean", 0.0),
            "model_rmse": self.meta.get("train_rmse", 0.0),
        }

    def get_status(self) -> dict:
        """현재 모델 상태를 반환합니다."""
        if self.pipeline is None:
            return {"trained": False, "message": "학습된 모델 없음"}
        return {"trained": True, **self.meta}

    def reset(self):
        """모델과 메타 파일을 삭제하고 초기화합니다."""
        self.pipeline = None
        self.meta = {}
        for path in [MODEL_PATH, META_PATH]:
            if os.path.exists(path):
                os.remove(path)
        log.info("모델 초기화 완료")

    # ── 내부 헬퍼 ──────────────────────────────────────────────────

    def _load_if_exists(self):
        """저장된 모델과 메타 파일이 있으면 로드합니다."""
        if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
            try:
                self.pipeline = joblib.load(MODEL_PATH)
                with open(META_PATH, encoding="utf-8") as f:
                    self.meta = json.load(f)
                log.info("저장된 모델 로드 완료 - %s (samples=%d, R²=%.4f)",
                         self.meta.get("model_type"), self.meta.get("sample_count", 0), self.meta.get("train_r2", 0))
            except Exception as e:
                log.warning("모델 로드 실패 (무시됨): %s", e)

    @staticmethod
    def _select_model(n: int):
        if n < 10:
            return LinearRegression()
        if n < 30:
            return GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

    @staticmethod
    def _to_matrix(features: List[dict]) -> np.ndarray:
        return np.array(
            [[f.get(k, 0.0) for k in FEATURE_KEYS] for f in features],
            dtype=float,
        )

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
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
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

_BASE_DIR = Path(os.getenv("ML_MODEL_DIR", str(Path(__file__).resolve().parent / "model")))
MODEL_PATH = str(_BASE_DIR / "ct_predictor.pkl")
META_PATH = str(_BASE_DIR / "ct_predictor_meta.json")
_VERSIONS_DIR = _BASE_DIR / "versions"
# 최대 보관 버전 수 (가장 오래된 것부터 자동 삭제).
MAX_VERSIONS = int(os.getenv("ML_MAX_MODEL_VERSIONS", "10"))

FEATURE_KEYS = ["band_intensity", "band_area", "relative_intensity", "band_width", "band_height", "log10_concentration"]


class ModelManager:
    """Ct값 예측 모델의 학습·저장·로드·예측을 담당합니다."""

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.meta: dict = {}
        # 학습은 독점 접근, 추론은 동시 허용. RLock으로 train 경계만 직렬화.
        self._train_lock = threading.Lock()
        self._load_if_exists()

    # ── 공개 메서드 ──────────────────────────────────────────────────

    def train(self, features: List[dict], ct_values: List[float]) -> dict:
        """
        훈련 데이터로 회귀 모델을 학습하고 디스크에 저장합니다.
        """
        with self._train_lock:
            X = self._to_matrix(features)
            y = np.array(ct_values, dtype=float)
            n = len(y)

            log.info("학습 시작 - 샘플: %d개, 특징: %s", n, FEATURE_KEYS)
            log.info("Ct 범위 - min=%.2f, max=%.2f, mean=%.2f", float(y.min()), float(y.max()), float(y.mean()))

            base_model = self._select_model(n)
            log.info("모델 선택: %s (샘플 수 %d 기준)", type(base_model).__name__, n)
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", base_model)])

            # 교차 검증
            cv = LeaveOneOut() if n <= 10 else min(5, n)
            cv_label = "LeaveOneOut" if n <= 10 else f"{cv}-fold"
            log.info("교차 검증 시작: %s", cv_label)
            cv_r2 = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
            log.info("교차 검증 완료: R² scores=%s", np.round(cv_r2, 4).tolist())

            # 전체 데이터로 최종 학습
            log.info("전체 데이터로 최종 모델 학습 중...")
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)

            cv_r2_clean = np.nan_to_num(cv_r2, nan=0.0)

            new_meta = {
                "sample_count": n,
                "model_type": type(base_model).__name__,
                "train_r2": round(float(r2_score(y, y_pred)), 4),
                "train_rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
                "cv_r2_mean": round(float(np.mean(cv_r2_clean)), 4),
                "cv_r2_std": round(float(np.std(cv_r2_clean)), 4),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_count": len(FEATURE_KEYS),
            }

            # 원자적 저장: 임시 파일 → rename
            _BASE_DIR.mkdir(parents=True, exist_ok=True)
            version_id = self._atomic_save(pipeline, new_meta)
            new_meta["version_id"] = version_id

            # 디스크 저장 성공 후 메모리 swap (in-flight predict는 이전 모델 사용)
            self.pipeline = pipeline
            self.meta = new_meta

            # 오래된 버전 정리
            self._prune_old_versions()

            log.info("모델 학습 완료 - %s | samples=%d, R²=%.4f, RMSE=%.4f, CV_R²=%.4f±%.4f",
                     new_meta["model_type"], n,
                     new_meta["train_r2"], new_meta["train_rmse"],
                     new_meta["cv_r2_mean"], new_meta["cv_r2_std"])
            return new_meta

    @staticmethod
    def _atomic_save(pipeline: Pipeline, meta: dict) -> str:
        """joblib dump + json dump을 동일 디렉토리 임시파일에 저장 후 os.replace.
        동시에 versions/{timestamp}/ 아래에도 스냅샷 보관. 반환값: version_id."""
        base_dir = _BASE_DIR
        with tempfile.NamedTemporaryFile(
            dir=str(base_dir), prefix=".tmp_model_", suffix=".pkl", delete=False
        ) as tmp_pkl:
            tmp_pkl_path = tmp_pkl.name
        with tempfile.NamedTemporaryFile(
            dir=str(base_dir), prefix=".tmp_meta_", suffix=".json", delete=False,
            mode="w", encoding="utf-8"
        ) as tmp_meta:
            tmp_meta_path = tmp_meta.name

        version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version_dir = _VERSIONS_DIR / version_id

        try:
            joblib.dump(pipeline, tmp_pkl_path)
            with open(tmp_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # 버전 아카이브 (current와 별개 복사본)
            _VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
            version_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(tmp_pkl_path, str(version_dir / "ct_predictor.pkl"))
            shutil.copy2(tmp_meta_path, str(version_dir / "ct_predictor_meta.json"))

            # current로 원자적 교체
            os.replace(tmp_pkl_path, MODEL_PATH)
            os.replace(tmp_meta_path, META_PATH)
            return version_id
        except Exception:
            for p in (tmp_pkl_path, tmp_meta_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            # 실패 시 스냅샷 정리
            try:
                if version_dir.exists():
                    import shutil
                    shutil.rmtree(str(version_dir), ignore_errors=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _prune_old_versions():
        """MAX_VERSIONS를 초과하는 가장 오래된 스냅샷을 삭제."""
        if not _VERSIONS_DIR.exists():
            return
        versions = sorted(
            [p for p in _VERSIONS_DIR.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        excess = len(versions) - MAX_VERSIONS
        if excess <= 0:
            return
        import shutil
        for old in versions[:excess]:
            try:
                shutil.rmtree(str(old), ignore_errors=True)
                log.info("오래된 모델 버전 삭제: %s", old.name)
            except OSError as e:
                log.warning("버전 삭제 실패 (%s): %s", old.name, e)

    def list_versions(self) -> List[dict]:
        """저장된 버전 목록 반환 (최신 순)."""
        if not _VERSIONS_DIR.exists():
            return []
        items = []
        for d in sorted(_VERSIONS_DIR.iterdir(), key=lambda p: p.name, reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "ct_predictor_meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except (IOError, OSError, json.JSONDecodeError) as e:
                    log.warning("버전 메타 읽기 실패 (%s): %s", d.name, e)
            is_current = meta.get("version_id") == self.meta.get("version_id")
            items.append({
                "version_id": d.name,
                "model_type": meta.get("model_type"),
                "sample_count": meta.get("sample_count"),
                "train_r2": meta.get("train_r2"),
                "cv_r2_mean": meta.get("cv_r2_mean"),
                "train_rmse": meta.get("train_rmse"),
                "trained_at": meta.get("trained_at"),
                "is_current": is_current,
            })
        return items

    def rollback(self, version_id: str) -> dict:
        """주어진 버전으로 현재 모델을 롤백."""
        # 경로 traversal 방지
        if not version_id or "/" in version_id or "\\" in version_id or ".." in version_id:
            raise ValueError("잘못된 version_id 형식입니다.")
        version_dir = _VERSIONS_DIR / version_id
        if not version_dir.is_dir():
            raise ValueError(f"해당 버전을 찾을 수 없습니다: {version_id}")

        src_pkl = version_dir / "ct_predictor.pkl"
        src_meta = version_dir / "ct_predictor_meta.json"
        if not src_pkl.exists() or not src_meta.exists():
            raise ValueError(f"버전 파일이 손상되었습니다: {version_id}")

        with self._train_lock:
            # 임시 파일 → os.replace 로 원자적 교체
            import shutil
            with tempfile.NamedTemporaryFile(
                dir=str(_BASE_DIR), prefix=".tmp_rollback_", suffix=".pkl", delete=False
            ) as tmp_pkl:
                tmp_pkl_path = tmp_pkl.name
            with tempfile.NamedTemporaryFile(
                dir=str(_BASE_DIR), prefix=".tmp_rollback_", suffix=".json", delete=False
            ) as tmp_meta:
                tmp_meta_path = tmp_meta.name
            try:
                shutil.copy2(str(src_pkl), tmp_pkl_path)
                shutil.copy2(str(src_meta), tmp_meta_path)
                os.replace(tmp_pkl_path, MODEL_PATH)
                os.replace(tmp_meta_path, META_PATH)
            except Exception:
                for p in (tmp_pkl_path, tmp_meta_path):
                    try:
                        if os.path.exists(p): os.remove(p)
                    except OSError:
                        pass
                raise

            # 메모리 reload
            self.pipeline = joblib.load(MODEL_PATH)
            with open(META_PATH, encoding="utf-8") as f:
                self.meta = json.load(f)
            log.info("모델 롤백 완료 → version=%s", version_id)
            return {"rolled_back_to": version_id, **self.meta}

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
        with self._train_lock:
            self.pipeline = None
            self.meta = {}
            for path in [MODEL_PATH, META_PATH]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError as e:
                        log.warning("모델 파일 삭제 실패 (%s): %s", path, e)
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
            except (IOError, OSError, json.JSONDecodeError) as e:
                log.warning("모델 파일 읽기 실패 (무시): %s", e)
            except Exception as e:
                log.warning("모델 역직렬화 실패 (무시): %s", e)

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

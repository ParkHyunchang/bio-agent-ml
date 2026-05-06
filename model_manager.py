"""
scikit-learn 기반 qPCR Ct값 예측 회귀 모델 관리 모듈.

샘플 수에 따라 적합한 모델을 자동 선택:
  < 10개  → LinearRegression (해석 가능, 과적합 방지)
  10~29개 → GradientBoostingRegressor (비선형, 중소 데이터)
  30개 이상 → RandomForestRegressor (앙상블, 충분한 데이터)

특징 벡터 (5차원, 이미지에서 추출한 밴드 특징만):
  [band_intensity, band_area, relative_intensity, band_width, band_height]

주의: log10_concentration은 레인 라벨에서 파생된 정답성 정보이므로
모델 입력으로 사용하지 않습니다 (data leakage 방지).
"""

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from logger import get_logger

log = get_logger("model_manager")

_BASE_DIR = Path(os.getenv("ML_MODEL_DIR", str(Path(__file__).resolve().parent / "model")))
MODEL_PATH = str(_BASE_DIR / "ct_predictor.pkl")
META_PATH = str(_BASE_DIR / "ct_predictor_meta.json")
_VERSIONS_DIR = _BASE_DIR / "versions"


def _parse_max_versions() -> int:
    """최대 보관 버전 수. 0/음수/비정수는 기본값으로 안전하게 폴백."""
    raw = os.getenv("ML_MAX_MODEL_VERSIONS", "10")
    try:
        n = int(raw)
    except ValueError:
        log.warning("ML_MAX_MODEL_VERSIONS=%r 가 정수가 아님 → 기본값 10 사용", raw)
        return 10
    if n < 1:
        log.warning("ML_MAX_MODEL_VERSIONS=%d 가 1 미만 → 1로 보정", n)
        return 1
    return n


def _parse_float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r 가 float가 아님 → 기본값 %s 사용", key, raw, default)
        return default


# 최대 보관 버전 수 (가장 오래된 것부터 자동 삭제).
MAX_VERSIONS = _parse_max_versions()

# Ct 유효 범위. predict()가 이 범위를 벗어나면 미검출로 처리.
CT_MIN = _parse_float_env("ML_CT_MIN", 5.0)
CT_MAX = _parse_float_env("ML_CT_MAX", 50.0)

FEATURE_KEYS = ["band_intensity", "band_area", "relative_intensity", "band_width", "band_height"]


class ModelManager:
    """Ct값 예측 모델의 학습·저장·로드·예측을 담당합니다."""

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.meta: dict = {}
        # 학습/롤백/리셋은 독점 접근, 추론은 락 없이 동시 허용
        # (predict는 self.pipeline/self.meta를 시작 시점에 한 번만 스냅샷하여 일관성 확보).
        self._train_lock = threading.Lock()
        self._load_if_exists()

    # ── 공개 메서드 ──────────────────────────────────────────────────

    def train(self, features: List[dict], ct_values: List[float]) -> dict:
        """
        훈련 데이터로 회귀 모델을 학습하고 디스크에 저장합니다.
        """
        with self._train_lock:
            # 입력 검증: 길이 불일치는 silent misalignment를 만들고, n<2면 CV/회귀가
            # 의미를 잃거나 sklearn에서 모호한 에러로 죽는다. 의도를 명확히 거절한다.
            if len(features) != len(ct_values):
                raise ValueError(
                    f"길이 불일치: features={len(features)}, ct_values={len(ct_values)}"
                )
            if len(ct_values) < 2:
                raise ValueError(
                    f"학습에는 최소 2개 샘플이 필요합니다 (현재 {len(ct_values)})"
                )

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

            # 데이터 품질 문제(예: 모든 y가 동일)로 인해 R²가 NaN 또는 극단적 음수가
            # 나올 수 있다. 메타에 그대로 노출되면 대시보드/클라이언트가 오해할 수 있어
            # 하한선을 둬서 안정화하고, 비정상 분포는 워닝으로 남긴다.
            CV_R2_FLOOR = -1.0
            nan_count = int(np.isnan(cv_r2).sum())
            below_floor = cv_r2[~np.isnan(cv_r2)] < CV_R2_FLOOR
            if nan_count > 0:
                log.warning("교차 검증 R²에 NaN %d개 포함 → 0.0으로 대체", nan_count)
            if below_floor.any():
                log.warning(
                    "교차 검증 R²가 하한(%g) 미만인 fold %d개 발견 (raw=%s) → 클리핑",
                    CV_R2_FLOOR, int(below_floor.sum()), np.round(cv_r2, 4).tolist(),
                )

            # 전체 데이터로 최종 학습
            log.info("전체 데이터로 최종 모델 학습 중...")
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)

            cv_r2_clean = np.clip(np.nan_to_num(cv_r2, nan=0.0), CV_R2_FLOOR, 1.0)

            new_meta = {
                "sample_count": n,
                "model_type": type(base_model).__name__,
                "train_r2": round(float(r2_score(y, y_pred)), 4),
                "train_rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
                "cv_r2_mean": round(float(np.mean(cv_r2_clean)), 4),
                "cv_r2_std": round(float(np.std(cv_r2_clean)), 4),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_count": len(FEATURE_KEYS),
                # 학습 시점의 피처 이름과 순서를 함께 저장한다. 차원은 같지만 순서/이름이
                # 바뀌면 dimension check만으로는 잡히지 않는 silent schema drift가 발생.
                "feature_keys": list(FEATURE_KEYS),
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
        동시에 versions/{timestamp}/ 아래에도 스냅샷 보관. 반환값: version_id.

        pkl/meta 두 파일 교체는 os.replace 두 번 호출이라 사이에서 실패하면 어긋날 수
        있다. 기존 current 파일을 .bak으로 백업하고, 두 번째 replace가 실패하면 양쪽
        모두 백업으로 되돌려 디스크 상태의 정합성을 보장한다.
        """
        base_dir = _BASE_DIR

        # mkstemp으로 fd를 받자마자 닫아 Windows에서 파일 핸들 잔류로 인한
        # PermissionError(공유 위반)를 방지한다. NamedTemporaryFile은 with 블록
        # 종료 시점이 OS별로 미묘하게 다를 수 있어 명시적으로 close를 분리.
        fd_pkl, tmp_pkl_path = tempfile.mkstemp(
            dir=str(base_dir), prefix=".tmp_model_", suffix=".pkl"
        )
        os.close(fd_pkl)
        fd_meta, tmp_meta_path = tempfile.mkstemp(
            dir=str(base_dir), prefix=".tmp_meta_", suffix=".json"
        )
        os.close(fd_meta)

        version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version_dir = _VERSIONS_DIR / version_id
        succeeded = False
        backup_pkl: str | None = None
        backup_meta: str | None = None

        try:
            joblib.dump(pipeline, tmp_pkl_path)
            with open(tmp_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # 버전 아카이브 (current와 별개 복사본)
            _VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
            version_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_pkl_path, str(version_dir / "ct_predictor.pkl"))
            shutil.copy2(tmp_meta_path, str(version_dir / "ct_predictor_meta.json"))

            # 기존 current 파일을 백업 (실패 시 복원용).
            if os.path.exists(MODEL_PATH):
                backup_pkl = MODEL_PATH + ".bak"
                shutil.copy2(MODEL_PATH, backup_pkl)
            if os.path.exists(META_PATH):
                backup_meta = META_PATH + ".bak"
                shutil.copy2(META_PATH, backup_meta)

            # current로 원자적 교체 (두 단계: pkl → meta).
            try:
                os.replace(tmp_pkl_path, MODEL_PATH)
                os.replace(tmp_meta_path, META_PATH)
            except Exception:
                # pkl만 교체된 상태일 수 있다. 양쪽을 백업으로 되돌려 정합성 회복.
                if backup_pkl and os.path.exists(backup_pkl):
                    try:
                        shutil.copy2(backup_pkl, MODEL_PATH)
                    except OSError as restore_err:
                        log.error("MODEL_PATH 복원 실패: %s", restore_err)
                if backup_meta and os.path.exists(backup_meta):
                    try:
                        shutil.copy2(backup_meta, META_PATH)
                    except OSError as restore_err:
                        log.error("META_PATH 복원 실패: %s", restore_err)
                raise

            succeeded = True
            return version_id
        except Exception as e:
            log.error("모델 저장 중 치명적 오류 발생: %s", e, exc_info=True)
            raise
        finally:
            # 성공 경로에서 tmp는 os.replace로 이미 소비됨. 실패 경로 + 백업은 여기서 정리.
            for p in (tmp_pkl_path, tmp_meta_path, backup_pkl, backup_meta):
                if not p:
                    continue
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError as cleanup_err:
                    log.warning("임시/백업 파일 정리 실패 (%s): %s", p, cleanup_err)
            # 실패 시 이번 회차에 만든 스냅샷 디렉토리 정리.
            if not succeeded and version_dir.exists():
                try:
                    shutil.rmtree(str(version_dir), ignore_errors=True)
                except OSError as cleanup_err:
                    log.warning("스냅샷 디렉토리 정리 실패 (%s): %s", version_dir, cleanup_err)

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
            # 1) 메모리에 먼저 로드해 손상 여부를 검증한다 (실패하면 디스크 변경 없음).
            try:
                new_pipeline = joblib.load(str(src_pkl))
                with open(src_meta, encoding="utf-8") as f:
                    new_meta = json.load(f)
            except Exception as e:
                log.error("롤백 대상 버전 로드 실패 (%s): %s", version_id, e)
                raise

            # 2) 디스크 current를 원자적으로 교체. _atomic_save와 동일한 백업-복원 패턴.
            backup_pkl: str | None = None
            backup_meta: str | None = None
            fd_pkl, tmp_pkl_path = tempfile.mkstemp(
                dir=str(_BASE_DIR), prefix=".tmp_rollback_", suffix=".pkl"
            )
            os.close(fd_pkl)
            fd_meta, tmp_meta_path = tempfile.mkstemp(
                dir=str(_BASE_DIR), prefix=".tmp_rollback_", suffix=".json"
            )
            os.close(fd_meta)
            try:
                shutil.copy2(str(src_pkl), tmp_pkl_path)
                shutil.copy2(str(src_meta), tmp_meta_path)
                if os.path.exists(MODEL_PATH):
                    backup_pkl = MODEL_PATH + ".bak"
                    shutil.copy2(MODEL_PATH, backup_pkl)
                if os.path.exists(META_PATH):
                    backup_meta = META_PATH + ".bak"
                    shutil.copy2(META_PATH, backup_meta)
                try:
                    os.replace(tmp_pkl_path, MODEL_PATH)
                    os.replace(tmp_meta_path, META_PATH)
                except Exception:
                    if backup_pkl and os.path.exists(backup_pkl):
                        try:
                            shutil.copy2(backup_pkl, MODEL_PATH)
                        except OSError as restore_err:
                            log.error("MODEL_PATH 복원 실패: %s", restore_err)
                    if backup_meta and os.path.exists(backup_meta):
                        try:
                            shutil.copy2(backup_meta, META_PATH)
                        except OSError as restore_err:
                            log.error("META_PATH 복원 실패: %s", restore_err)
                    raise
            finally:
                for p in (tmp_pkl_path, tmp_meta_path, backup_pkl, backup_meta):
                    if not p:
                        continue
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass

            # 3) 디스크 교체 성공 후 메모리 swap (1단계에서 이미 검증된 객체 재사용).
            self.pipeline = new_pipeline
            self.meta = new_meta
            log.info("모델 롤백 완료 → version=%s", version_id)
            return {"rolled_back_to": version_id, **self.meta}

    def predict(self, features: dict) -> dict:
        """
        단일 특징 dict로 Ct값을 예측합니다.

        Returns:
            dict: predicted_ct, model_r2, model_rmse
        """
        # train과 predict가 동시에 일어나는 경우 pipeline/meta가 부분적으로 교체된
        # 상태에서 읽으면 피처 차원 불일치가 발생할 수 있다. 시작 시점에 한 번만
        # 스냅샷을 떠서 일관된 한 쌍으로 사용한다.
        current_pipeline = self.pipeline
        current_meta = self.meta

        if current_pipeline is None:
            raise ValueError("학습된 모델이 없습니다. /train 을 먼저 실행하세요.")

        X = self._to_matrix([features])
        # 차원 불일치는 silent truncate 대신 명시적 실패 — 잘못된 모델/피처 정의로 인한
        # 무의미한 예측이 클라이언트로 흘러가는 것을 차단한다.
        trained_feature_count = current_meta.get("feature_count", len(FEATURE_KEYS))
        if X.shape[1] != trained_feature_count:
            raise ValueError(
                f"피처 차원 불일치: 입력={X.shape[1]}, 학습={trained_feature_count}. "
                "모델 재학습이 필요합니다."
            )
        # 피처 이름/순서가 바뀐 모델은 차원이 같아도 의미가 완전히 다르다 (silent schema
        # drift). 메타에 feature_keys가 있다면 정확히 일치하는지 확인한다.
        # 구버전 모델은 feature_keys가 없을 수 있어 누락 시에는 통과시킨다 (호환성).
        trained_keys = current_meta.get("feature_keys")
        if trained_keys is not None and list(trained_keys) != list(FEATURE_KEYS):
            raise ValueError(
                f"피처 스키마 불일치: trained={list(trained_keys)}, "
                f"current={list(FEATURE_KEYS)}. 모델 재학습이 필요합니다."
            )
        predicted_ct = float(current_pipeline.predict(X)[0])

        if predicted_ct < CT_MIN or predicted_ct > CT_MAX:
            log.warning("Ct값 범위 이탈 (%.2f) → 미검출 처리", predicted_ct)
            raise ValueError(f"예측값({predicted_ct:.2f})이 유효 범위({CT_MIN}–{CT_MAX})를 벗어났습니다.")

        log.info("Ct값 예측 완료 - predicted_ct=%.2f (model=%s)", predicted_ct, current_meta.get("model_type"))
        # 메타가 비어 있으면 0.0 대신 None을 반환 — 클라이언트가 "지표 없음"과 "지표가
        # 0이다"를 구분할 수 있도록 한다.
        return {
            "predicted_ct": round(predicted_ct, 2),
            "model_r2": current_meta.get("cv_r2_mean"),
            "model_rmse": current_meta.get("train_rmse"),
        }

    def get_status(self) -> dict:
        """현재 모델 상태를 반환합니다."""
        # predict와 동일한 이유로 한 번에 스냅샷. 또한 호출자가 반환된 dict를 변형해도
        # self.meta가 오염되지 않도록 shallow copy로 분리한다.
        current_pipeline = self.pipeline
        current_meta = self.meta
        if current_pipeline is None:
            return {"trained": False, "message": "학습된 모델 없음"}
        return {"trained": True, **current_meta.copy()}

    def reset(self):
        """모델, 메타, 그리고 모든 버전 스냅샷을 삭제하고 초기화합니다."""
        with self._train_lock:
            self.pipeline = None
            self.meta = {}
            for path in [MODEL_PATH, META_PATH]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError as e:
                        log.warning("모델 파일 삭제 실패 (%s): %s", path, e)
            # 버전 아카이브도 함께 정리 — reset 후 list_versions/rollback 결과가
            # 비어 있도록 일관성을 맞춘다.
            if _VERSIONS_DIR.exists():
                try:
                    shutil.rmtree(str(_VERSIONS_DIR), ignore_errors=True)
                except OSError as e:
                    log.warning("버전 디렉토리 삭제 실패 (%s): %s", _VERSIONS_DIR, e)
            log.info("모델 초기화 완료")

    # ── 내부 헬퍼 ──────────────────────────────────────────────────

    def _load_if_exists(self):
        """저장된 모델과 메타 파일이 있으면 로드합니다.

        pipeline 로드는 성공했는데 meta 로드가 실패하는 경우 self.pipeline만 채워지면
        cv_r2/version_id 등이 빠진 채 predict가 잘못된 메타를 반환한다. 둘 다 성공할
        때만 한꺼번에 self에 반영한다 (all-or-nothing).
        """
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            return
        try:
            pipeline = joblib.load(MODEL_PATH)
            with open(META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
        except (IOError, OSError, json.JSONDecodeError) as e:
            log.warning("모델 파일 읽기 실패 (무시): %s", e)
            return
        except Exception as e:
            log.warning("모델 역직렬화 실패 (무시): %s", e)
            return

        self.pipeline = pipeline
        self.meta = meta
        log.info("저장된 모델 로드 완료 - %s (samples=%d, R²=%.4f)",
                 meta.get("model_type"), meta.get("sample_count", 0), meta.get("train_r2", 0))

    @staticmethod
    def _select_model(n: int):
        if n < 10:
            return LinearRegression()
        if n < 30:
            return GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

    @staticmethod
    def _to_matrix(features: List[dict]) -> np.ndarray:
        # 피처는 모두 물리적 치수/밝기이므로 0.0 fallback은 위험 (왜곡된 예측 유발).
        # 누락/타입 불일치는 침묵하지 말고 명시적으로 실패시킨다.
        matrix = []
        for idx, f in enumerate(features):
            row = []
            for k in FEATURE_KEYS:
                val = f.get(k)
                if val is None:
                    raise ValueError(f"샘플 index {idx}에서 필수 특징 '{k}'이(가) 누락되었습니다.")
                try:
                    row.append(float(val))
                except (ValueError, TypeError):
                    raise ValueError(
                        f"샘플 index {idx}의 '{k}' 값({val!r})을 float로 변환할 수 없습니다."
                    )
            matrix.append(row)
        return np.array(matrix, dtype=float)

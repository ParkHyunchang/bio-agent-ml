"""
PCR 젤 이미지 분석 ML 마이크로서비스.

엔드포인트:
  POST /extract        이미지 → 밴드 특징 JSON 반환
  POST /train          훈련 데이터 (features + ct_values) → 모델 학습
  POST /predict        새 이미지 → Ct값 예측
  GET  /model/status   현재 모델 정보 조회
  GET  /health         헬스체크
"""

import asyncio
import os
import random
import time
from contextlib import asynccontextmanager

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from image_processor import extract_gel_lanes, process_gel_image
from logger import get_logger, setup_logging
from model_manager import ModelManager

setup_logging()
log = get_logger("main")

# 재현성: 환경변수로 설정된 seed를 전역 적용 (sklearn 학습·cross validation 결정론성 향상).
_SEED = int(os.getenv("ML_RANDOM_SEED", "42"))
random.seed(_SEED)
np.random.seed(_SEED)

# ── 설정 ────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = int(os.getenv("ML_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # 10MB
ALLOWED_IMAGE_MIME = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/bmp", "image/tiff",
}
MAX_N_LANES = 20
MIN_N_LANES = 1
REQUEST_TIMEOUT_S = float(os.getenv("ML_REQUEST_TIMEOUT_S", "120"))

_cors_origins_env = os.getenv("ML_CORS_ORIGINS", "http://localhost:3211,http://127.0.0.1:3211").strip()
CORS_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("PCR Gel ML Service 시작 (port=3212, cors_origins=%s)", CORS_ORIGINS)
    yield


app = FastAPI(
    title="PCR Gel ML Service",
    description="PCR 젤 이미지에서 qPCR Ct값을 예측하는 회귀 모델 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def log_and_timeout(request: Request, call_next):
    start = time.perf_counter()
    req_id = request.headers.get("X-Request-ID", "-")
    try:
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.warning("[%s] %s %s → TIMEOUT (%.1fms, limit=%.0fs)",
                    req_id, request.method, request.url.path, elapsed_ms, REQUEST_TIMEOUT_S)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=504,
            content={"error": "처리 시간이 제한을 초과했습니다."},
            headers={"X-Request-ID": req_id} if req_id != "-" else {},
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info("[%s] %s %s → %d (%.1fms)",
             req_id, request.method, request.url.path, response.status_code, elapsed_ms)
    if req_id != "-":
        response.headers["X-Request-ID"] = req_id
    return response


# 앱 시작 시 저장된 모델 자동 로드
model_manager = ModelManager()


# ── 요청/응답 스키마 ───────────────────────────────────────────────

class TrainRequest(BaseModel):
    features: List[dict] = Field(..., description="각 이미지에서 추출한 특징 dict 리스트", min_length=3)
    ct_values: List[float] = Field(..., description="대응하는 실측 Ct값 리스트", min_length=3)


# ── 업로드 검증 헬퍼 ────────────────────────────────────────────

async def _read_and_validate_image(file: UploadFile) -> bytes:
    """업로드 파일 MIME/크기 검증 후 바이트 반환."""
    if file.content_type and file.content_type.lower() not in ALLOWED_IMAGE_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"허용되지 않은 이미지 형식입니다: {file.content_type}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"이미지 크기가 제한을 초과했습니다 (최대 {MAX_IMAGE_BYTES // (1024 * 1024)}MB).",
        )
    return image_bytes


def _validate_n_lanes(n_lanes: int) -> int:
    if n_lanes < MIN_N_LANES or n_lanes > MAX_N_LANES:
        raise HTTPException(
            status_code=400,
            detail=f"n_lanes는 {MIN_N_LANES}~{MAX_N_LANES} 사이여야 합니다.",
        )
    return n_lanes


# ── 엔드포인트 ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    status = model_manager.get_status()
    return {
        "status": "ok",
        "model_trained": bool(status.get("trained", False)),
        "model_type": status.get("model_type"),
    }


@app.post("/extract", summary="이미지에서 밴드 특징 추출")
async def extract_features(file: UploadFile = File(...)):
    """
    PCR 젤 이미지를 받아 OpenCV로 밴드 특징을 추출합니다.
    """
    image_bytes = await _read_and_validate_image(file)
    try:
        return await asyncio.to_thread(process_gel_image, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (IOError, OSError) as e:
        log.error("이미지 I/O 오류: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")
    except Exception as e:
        log.error("이미지 처리 예기치 못한 오류: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")


@app.post("/train", summary="회귀 모델 학습")
async def train_model(request: TrainRequest):
    """
    훈련 데이터(features + ct_values)로 회귀 모델을 학습합니다.
    """
    if len(request.features) != len(request.ct_values):
        raise HTTPException(
            status_code=400,
            detail=f"features({len(request.features)}개)와 ct_values({len(request.ct_values)}개) 개수가 일치해야 합니다.",
        )
    try:
        return await asyncio.to_thread(model_manager.train, request.features, request.ct_values)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("모델 학습 오류: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="모델 학습 중 오류가 발생했습니다.")


@app.post("/predict", summary="새 이미지로 Ct값 예측")
async def predict_ct(file: UploadFile = File(...)):
    """
    새 PCR 젤 이미지를 받아 학습된 모델로 Ct값을 예측합니다.
    """
    image_bytes = await _read_and_validate_image(file)
    try:
        log.info("단일 Ct 예측 시작: file=%s, size=%dbytes", file.filename, len(image_bytes))
        features = await asyncio.to_thread(process_gel_image, image_bytes)
        prediction = await asyncio.to_thread(model_manager.predict, features)
        prediction["features"] = features
        log.info("단일 Ct 예측 완료: file=%s, predicted_ct=%.2f, model_r2=%.4f",
                 file.filename, prediction.get("predicted_ct", 0), prediction.get("model_r2", 0))
        return prediction
    except ValueError as e:
        log.warning("Ct 예측 범위 이탈 또는 미학습: file=%s, reason=%s", file.filename, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("Ct 예측 오류: file=%s", file.filename, exc_info=True)
        raise HTTPException(status_code=500, detail="예측 중 오류가 발생했습니다.")


@app.post("/extract-gel", summary="젤 이미지 전체 레인 특징 추출")
async def extract_gel_endpoint(
    file: UploadFile = File(...),
    n_lanes: int = Form(10),
):
    """
    PCR 젤 이미지의 각 레인에서 밴드 특징을 추출합니다.
    """
    n_lanes = _validate_n_lanes(n_lanes)
    image_bytes = await _read_and_validate_image(file)
    try:
        log.info("레인 추출 시작: file=%s, n_lanes=%d", file.filename, n_lanes)
        result = await asyncio.to_thread(extract_gel_lanes, image_bytes, n_lanes)
        log.info("레인 추출 완료: file=%s, 검출=%d/%d개",
                 file.filename, result["n_lanes_detected"], n_lanes)
        return result
    except ValueError as e:
        log.warning("레인 추출 오류: file=%s, reason=%s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("레인 추출 오류: file=%s", file.filename, exc_info=True)
        raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")


@app.post("/predict-gel", summary="젤 이미지 전체 레인 Ct값 예측")
async def predict_gel_endpoint(
    file: UploadFile = File(...),
    n_lanes: int = Form(10),
):
    """
    PCR 젤 이미지의 각 레인에서 Ct값을 예측합니다.
    """
    n_lanes = _validate_n_lanes(n_lanes)
    image_bytes = await _read_and_validate_image(file)
    try:
        log.info("멀티레인 예측 시작: file=%s, n_lanes=%d", file.filename, n_lanes)
        lanes_data = await asyncio.to_thread(extract_gel_lanes, image_bytes, n_lanes)
        predicted_count = 0
        for lane in lanes_data["lanes"]:
            try:
                pred = await asyncio.to_thread(model_manager.predict, lane)
                predicted_count += 1
            except ValueError:
                pred = {"predicted_ct": None, "model_r2": None, "model_rmse": None}
            lane.update(pred)

        lanes_data["model_trained"] = model_manager.pipeline is not None
        detected = sum(
            1 for l in lanes_data["lanes"]
            if not l.get("is_negative", True) and l.get("label") not in ("M", "NTC")
        )
        log.info("멀티레인 예측 완료: file=%s, 전체=%d개, 검출=%d개, 예측성공=%d개",
                 file.filename, len(lanes_data["lanes"]), detected, predicted_count)
        return lanes_data
    except ValueError as e:
        log.warning("멀티레인 예측 오류: file=%s, reason=%s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("멀티레인 예측 오류: file=%s", file.filename, exc_info=True)
        raise HTTPException(status_code=500, detail="예측 중 오류가 발생했습니다.")


@app.get("/model/status", summary="모델 현황 조회")
def model_status():
    return model_manager.get_status()


@app.delete("/model", summary="학습 모델 초기화")
def reset_model():
    model_manager.reset()
    return {"status": "reset"}


@app.get("/model/versions", summary="저장된 모델 버전 목록")
def list_model_versions():
    return {"versions": model_manager.list_versions()}


@app.post("/model/rollback", summary="특정 버전으로 모델 롤백")
async def rollback_model(payload: dict):
    version_id = (payload or {}).get("version_id")
    if not version_id or not isinstance(version_id, str):
        raise HTTPException(status_code=400, detail="version_id가 필요합니다.")
    try:
        return await asyncio.to_thread(model_manager.rollback, version_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("모델 롤백 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="모델 롤백 중 오류가 발생했습니다.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3212)

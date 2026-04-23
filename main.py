"""
PCR 젤 이미지 분석 ML 마이크로서비스.

엔드포인트:
  POST /extract        이미지 → 밴드 특징 JSON 반환
  POST /train          훈련 데이터 (features + ct_values) → 모델 학습
  POST /predict        새 이미지 → Ct값 예측
  GET  /model/status   현재 모델 정보 조회
  GET  /health         헬스체크
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from image_processor import extract_gel_lanes, process_gel_image
from logger import get_logger, setup_logging
from model_manager import ModelManager

setup_logging()
log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("PCR Gel ML Service 시작 (port=3212)")
    yield


app = FastAPI(
    title="PCR Gel ML Service",
    description="PCR 젤 이미지에서 qPCR Ct값을 예측하는 회귀 모델 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info("%s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# 앱 시작 시 저장된 모델 자동 로드
model_manager = ModelManager()


# ── 요청/응답 스키마 ───────────────────────────────────────────────

class TrainRequest(BaseModel):
    features: List[dict] = Field(..., description="각 이미지에서 추출한 특징 dict 리스트")
    ct_values: List[float] = Field(..., description="대응하는 실측 Ct값 리스트")


# ── 엔드포인트 ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", summary="이미지에서 밴드 특징 추출")
async def extract_features(file: UploadFile = File(...)):
    """
    PCR 젤 이미지를 받아 OpenCV로 밴드 특징을 추출합니다.

    - **band_intensity**: 밴드 평균 픽셀 밝기 (0–255)
    - **band_area**: 밴드 면적 (픽셀²)
    - **relative_intensity**: 이미지 최대 밝기 대비 상대값 (0–1)
    - **band_width**: 밴드 수평 너비 (픽셀)
    - **lanes_detected**: 검출된 밴드 수
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        return process_gel_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 오류: {e}")


@app.post("/train", summary="회귀 모델 학습")
def train_model(request: TrainRequest):
    """
    훈련 데이터(features + ct_values)로 회귀 모델을 학습합니다.
    샘플 수에 따라 LinearRegression / GradientBoosting / RandomForest 중 자동 선택.
    학습 후 model/ 디렉토리에 모델 파일이 저장됩니다.
    """
    if len(request.features) != len(request.ct_values):
        raise HTTPException(
            status_code=400,
            detail=f"features({len(request.features)}개)와 ct_values({len(request.ct_values)}개) 개수가 일치해야 합니다.",
        )
    if len(request.features) < 3:
        raise HTTPException(
            status_code=400,
            detail="최소 3개 이상의 훈련 데이터가 필요합니다.",
        )
    try:
        return model_manager.train(request.features, request.ct_values)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 학습 오류: {e}")


@app.post("/predict", summary="새 이미지로 Ct값 예측")
async def predict_ct(file: UploadFile = File(...)):
    """
    새 PCR 젤 이미지를 받아 학습된 모델로 Ct값을 예측합니다.
    모델이 학습되지 않은 경우 422 오류를 반환합니다.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        log.info("단일 Ct 예측 시작: file=%s, size=%dbytes", file.filename, len(image_bytes))
        features = process_gel_image(image_bytes)
        prediction = model_manager.predict(features)
        prediction["features"] = features
        log.info("단일 Ct 예측 완료: file=%s, predicted_ct=%.2f, model_r2=%.4f",
                 file.filename, prediction.get("predicted_ct", 0), prediction.get("model_r2", 0))
        return prediction
    except ValueError as e:
        log.warning("Ct 예측 범위 이탈: file=%s, reason=%s", file.filename, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("Ct 예측 오류: file=%s, error=%s", file.filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 오류: {e}")


@app.post("/extract-gel", summary="젤 이미지 전체 레인 특징 추출")
async def extract_gel_endpoint(
    file: UploadFile = File(...),
    n_lanes: int = Form(10),
):
    """
    PCR 젤 이미지의 각 레인에서 밴드 특징을 추출합니다.
    레인 순서: M, 10^8, 10^7, 10^6, 10^5, 10^4, 10^3, 10^2, 10^1, NTC
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        log.info("레인 추출 시작: file=%s, n_lanes=%d", file.filename, n_lanes)
        result = extract_gel_lanes(image_bytes, n_lanes)
        log.info("레인 추출 완료: file=%s, 검출=%d/%d개",
                 file.filename, result["n_lanes_detected"], n_lanes)
        return result
    except ValueError as e:
        log.warning("레인 추출 오류: file=%s, reason=%s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("레인 추출 오류: file=%s, error=%s", file.filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"이미지 처리 오류: {e}")


@app.post("/predict-gel", summary="젤 이미지 전체 레인 Ct값 예측")
async def predict_gel_endpoint(
    file: UploadFile = File(...),
    n_lanes: int = Form(10),
):
    """
    PCR 젤 이미지의 각 레인에서 Ct값을 예측합니다.
    모델이 학습되지 않은 경우 predicted_ct는 null로 반환됩니다.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        log.info("멀티레인 예측 시작: file=%s, n_lanes=%d", file.filename, n_lanes)
        lanes_data = extract_gel_lanes(image_bytes, n_lanes)
        predicted_count = 0
        for lane in lanes_data["lanes"]:
            try:
                pred = model_manager.predict(lane)
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
        log.error("멀티레인 예측 오류: file=%s, error=%s", file.filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 오류: {e}")


@app.get("/model/status", summary="모델 현황 조회")
def model_status():
    """현재 학습된 모델의 메타 정보를 반환합니다 (R², RMSE, 샘플 수 등)."""
    return model_manager.get_status()


@app.delete("/model", summary="학습 모델 초기화")
def reset_model():
    """모델 파일을 삭제하고 미학습 상태로 초기화합니다."""
    model_manager.reset()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3212)

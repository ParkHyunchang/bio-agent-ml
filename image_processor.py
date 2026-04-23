"""
PCR 젤 전기영동 이미지 처리 파사드.

단일 밴드 추출 (하위 호환):
  process_gel_image() → 전체 이미지에서 가장 큰 밴드 1개 반환

멀티레인 추출:
  extract_gel_lanes() → 레인별 밴드 특징 리스트 반환
  레인 순서: M, 10^8, 10^7, 10^6, 10^5, 10^4, 10^3, 10^2, 10^1, NTC
"""

import cv2
import numpy as np

from lane_detector import detect_lanes
from feature_extractor import extract_lane_features, LANE_LABELS
from logger import get_logger

log = get_logger("image_processor")


def extract_gel_lanes(image_bytes: bytes, n_lanes: int = 10) -> dict:
    """
    PCR 젤 이미지에서 각 레인의 밴드 특징을 추출합니다.

    Returns:
        dict:
            lanes              list[dict]  레인별 특징값 리스트
            n_lanes_detected   int         실제 검출된 레인 수
            warning            str|None    경고 메시지
    """
    log.info("멀티레인 추출 시작: size=%dbytes, 목표=%d레인", len(image_bytes), n_lanes)
    ctx = detect_lanes(image_bytes, n_lanes)
    img = ctx["image"]
    lane_centers = ctx["lane_centers"]
    global_max = ctx["global_max"]
    roi_top = ctx["roi_top"]
    roi_bot = ctx["roi_bot"]
    half_w = ctx["half_w"]
    w = img.shape[1]

    labels = LANE_LABELS[:n_lanes]
    lanes = []
    for idx, center in enumerate(lane_centers):
        label = labels[idx] if idx < len(labels) else f"Lane_{idx}"
        col_start = max(0, center - half_w)
        col_end = min(w, center + half_w)
        lane_img = img[roi_top:roi_bot, col_start:col_end]
        lanes.append(extract_lane_features(lane_img, label, idx, global_max))

    saturated = sum(1 for l in lanes if l.get("is_saturated"))
    negative  = sum(1 for l in lanes if l.get("is_negative"))
    log.info("멀티레인 추출 완료: %d개 레인 (포화=%d, 미검출=%d)", len(lanes), saturated, negative)
    return {
        "lanes": lanes,
        "n_lanes_detected": len(lanes),
        "warning": None if len(lanes) == n_lanes else f"레인 {len(lanes)}개 검출 (예상 {n_lanes}개)",
    }


def process_gel_image(image_bytes: bytes) -> dict:
    """
    PCR 젤 이미지 바이트에서 밴드 특징을 추출합니다. (단일 밴드, 하위 호환)

    Returns:
        dict:
            band_intensity      (float) 밴드 영역 평균 픽셀 밝기 (0–255)
            band_area           (float) 밴드 픽셀 면적 (px²)
            relative_intensity  (float) 이미지 최대 밝기 대비 상대값 (0–1)
            band_width          (float) 밴드 수평 너비 (px)
            band_height         (float) 밴드 수직 높이 (px)
            lanes_detected      (int)   검출된 밴드(컨투어) 수
            warning             (str)   경고 메시지 (밴드 검출 실패 시)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다. 지원 형식: JPEG, PNG, BMP, TIFF")

    mean_brightness = float(np.mean(img))
    if mean_brightness > 128:
        img = cv2.bitwise_not(img)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        manual_thresh = mean_brightness + float(np.std(blurred))
        _, thresh2 = cv2.threshold(blurred, int(manual_thresh), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        log.info("밴드 검출 실패 - 전체 이미지 통계 사용 (size=%dx%d)", img.shape[1], img.shape[0])
        return {
            "band_intensity": float(np.mean(blurred)),
            "band_area": float(blurred.size),
            "relative_intensity": float(np.mean(blurred)) / 255.0,
            "band_width": float(blurred.shape[1]),
            "band_height": float(blurred.shape[0]),
            "lanes_detected": 0,
            "warning": "밴드 자동 검출 실패, 전체 이미지 통계를 사용합니다.",
        }

    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(blurred)
    cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

    band_pixels = blurred[mask > 0]
    band_intensity = float(np.mean(band_pixels)) if band_pixels.size > 0 else 0.0
    band_area = float(cv2.contourArea(main_contour))
    _, _, band_width, band_height = cv2.boundingRect(main_contour)

    global_max = float(blurred.max())
    relative_intensity = band_intensity / global_max if global_max > 0 else 0.0

    result = {
        "band_intensity": band_intensity,
        "band_area": band_area,
        "relative_intensity": relative_intensity,
        "band_width": float(band_width),
        "band_height": float(band_height),
        "lanes_detected": len(contours),
    }

    if len(contours) == 1:
        result["warning"] = "밴드 1개만 검출됨. 래더 포함 여부를 확인하세요."
        log.info("밴드 1개 검출 - 래더 확인 필요 (intensity=%.1f, area=%.0f)", band_intensity, band_area)
    else:
        log.info("밴드 검출 완료 (lanes=%d, intensity=%.1f, area=%.0f, rel=%.3f)",
                 len(contours), band_intensity, band_area, relative_intensity)

    return result

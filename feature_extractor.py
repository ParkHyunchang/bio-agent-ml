"""
PCR 젤 레인 슬라이스에서 밴드 특징을 추출하는 모듈.

주요 함수:
  extract_lane_features() → 단일 레인 이미지에서 밴드 강도·면적·너비 등 특징 반환
"""

import cv2
import numpy as np

from logger import get_logger

log = get_logger("feature_extractor")

LANE_LABELS = ["M", "10^8", "10^7", "10^6", "10^5", "10^4", "10^3", "10^2", "10^1", "NTC"]
LOG10_CONC = {
    "10^8": 8, "10^7": 7, "10^6": 6, "10^5": 5,
    "10^4": 4, "10^3": 3, "10^2": 2, "10^1": 1,
}

# Faint band: 검출은 됐지만 global_max 대비 약한 밴드.
# is_negative(< 1%) 위, 정상 밴드(>= 10%) 아래 구간 → 1% ≤ relative_intensity < 10%.
FAINT_INTENSITY_RATIO = 0.10


def extract_lane_features(lane_img: np.ndarray, label: str,
                           lane_index: int, global_max: float) -> dict:
    """
    단일 레인 슬라이스에서 밴드 특징을 추출합니다.

    Args:
        lane_img:    레인 영역 그레이스케일 이미지 슬라이스
        label:       레인 레이블 (예: "10^5", "NTC")
        lane_index:  레인 순서 인덱스
        global_max:  전체 이미지 최대 픽셀 강도 (상대 강도 계산용)

    Returns:
        dict: band_intensity, band_area, relative_intensity, band_width,
              band_height, is_saturated, is_negative, is_primer_dimer 포함
    """
    h, w = lane_img.shape
    base = {
        "lane_index": lane_index,
        "label": label,
        "log10_concentration": LOG10_CONC.get(label),
    }

    if h == 0 or w == 0:
        return {**base, "band_intensity": 0.0, "band_area": 0.0,
                "relative_intensity": 0.0, "band_width": 0.0, "band_height": 0.0,
                "is_saturated": False, "is_negative": True,
                "is_faint": False, "is_primer_dimer": False}

    row_profile = np.mean(lane_img, axis=1)
    band_row = int(np.argmax(row_profile))
    peak_intensity = float(row_profile[band_row])

    # 전체 이미지 최대값 대비 1% 미만이면 노이즈로 판단 (고정 임계값 대신 상대 기준 사용)
    noise_threshold = max(1.0, global_max * 0.01)
    is_negative = peak_intensity < noise_threshold
    is_saturated = peak_intensity > 240.0

    if is_negative:
        return {**base, "band_intensity": 0.0, "band_area": 0.0,
                "relative_intensity": 0.0, "band_width": 0.0, "band_height": 0.0,
                "is_saturated": False, "is_negative": True,
                "is_faint": False, "is_primer_dimer": False}

    # 밴드 행 ±15% 범위에서 컨투어 탐색
    half_band = max(5, int(h * 0.15))
    row_start = max(0, band_row - half_band)
    row_end = min(h, band_row + half_band + 1)
    band_region = lane_img[row_start:row_end, :]

    thresh_val = max(1, int(peak_intensity * 0.30))
    _, bin_mask = cv2.threshold(band_region, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        band_intensity = float(np.mean(band_region[band_region > thresh_val])) \
            if np.any(band_region > thresh_val) else float(np.mean(band_region))
        band_area = float(np.sum(band_region > thresh_val))
        bw, bh = int(w), int(row_end - row_start)
    else:
        main_c = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(band_region)
        cv2.drawContours(mask, [main_c], -1, 255, thickness=cv2.FILLED)
        pixels = band_region[mask > 0]
        band_intensity = float(np.mean(pixels)) if pixels.size > 0 else 0.0
        band_area = float(cv2.contourArea(main_c))
        _, _, bw, bh = cv2.boundingRect(main_c)

    relative_intensity = band_intensity / global_max if global_max > 0 else 0.0

    # Faint: 노이즈(< 1%)는 넘었지만 정상 밴드(>= 10%)에 못 미치는 구간. 포화된 밴드는 제외.
    is_faint = (not is_saturated) and (relative_intensity < FAINT_INTENSITY_RATIO)

    # 프라이머 다이머: 젤 하단 35% + 높이가 ROI의 8% 미만인 얇은 밴드
    band_position_ratio = band_row / max(h, 1)
    is_primer_dimer = (
        band_position_ratio > 0.65
        and bh < h * 0.08
        and not is_saturated
    )

    log.debug("레인 %d (%s): intensity=%.1f area=%.0f rel=%.3f sat=%s faint=%s",
              lane_index, label, band_intensity, band_area,
              relative_intensity, is_saturated, is_faint)

    return {
        **base,
        "band_intensity": round(band_intensity, 2),
        "band_area": round(band_area, 2),
        "relative_intensity": round(relative_intensity, 4),
        "band_width": float(bw),
        "band_height": float(bh),
        "is_saturated": bool(is_saturated),
        "is_negative": False,
        "is_faint": bool(is_faint),
        "is_primer_dimer": bool(is_primer_dimer),
    }

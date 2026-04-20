"""
PCR 젤 전기영동 이미지에서 밴드 특징을 추출하는 모듈.

단일 밴드 추출 (하위 호환):
  process_gel_image() → 전체 이미지에서 가장 큰 밴드 1개 반환

멀티레인 추출 (신규):
  extract_gel_lanes() → 레인별 밴드 특징 리스트 반환
  레인 순서: M, 10^8, 10^7, 10^6, 10^5, 10^4, 10^3, 10^2, 10^1, NTC
"""

import cv2
import numpy as np

from logger import get_logger

log = get_logger("image_processor")

LANE_LABELS = ["M", "10^8", "10^7", "10^6", "10^5", "10^4", "10^3", "10^2", "10^1", "NTC"]
LOG10_CONC = {"10^8": 8, "10^7": 7, "10^6": 6, "10^5": 5,
              "10^4": 4, "10^3": 3, "10^2": 2, "10^1": 1}


# ── 멀티레인 추출 ─────────────────────────────────────────────────────

def extract_gel_lanes(image_bytes: bytes, n_lanes: int = 10) -> dict:
    """
    PCR 젤 이미지에서 각 레인의 밴드 특징을 추출합니다.

    Args:
        image_bytes: 이미지 파일 바이트
        n_lanes:     검출할 레인 수 (기본 10: M + 8희석 + NTC)

    Returns:
        dict:
            lanes              list[dict]  레인별 특징값 리스트
            n_lanes_detected   int         실제 검출된 레인 수
            warning            str|None    경고 메시지
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다. 지원 형식: JPEG, PNG, BMP, TIFF")

    h, w = img.shape

    # 1. 방향 정규화
    if float(np.mean(img)) > 128:
        img = cv2.bitwise_not(img)

    # 2. 노이즈 제거
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    global_max = float(blurred.max()) or 1.0

    # 3. 밴드 ROI (상단 텍스트 10%, 하단 텍스트 8% 제외)
    roi_top = int(h * 0.10)
    roi_bot = int(h * 0.92)
    roi = blurred[roi_top:roi_bot, :]

    # 4. 열 방향 강도 합산 → 레인 중심 검출
    col_profile = np.mean(roi, axis=0).astype(float)
    lane_centers = _find_lane_centers(col_profile, n_lanes)

    # 5. 레인 너비 결정
    half_w = max(w // (n_lanes * 2), 20)

    # 6. 각 레인에서 특징 추출
    labels = LANE_LABELS[:n_lanes]
    lanes = []
    for idx, center in enumerate(lane_centers):
        label = labels[idx] if idx < len(labels) else f"Lane_{idx}"
        col_start = max(0, center - half_w)
        col_end = min(w, center + half_w)

        lane_img = blurred[roi_top:roi_bot, col_start:col_end]
        feat = _extract_lane_features(lane_img, label, idx, global_max)
        lanes.append(feat)
        log.debug("레인 %d (%s): intensity=%.1f area=%.0f sat=%s neg=%s",
                  idx, label, feat["band_intensity"], feat["band_area"],
                  feat["is_saturated"], feat["is_negative"])

    log.info("멀티레인 추출 완료: %d개 레인", len(lanes))
    return {
        "lanes": lanes,
        "n_lanes_detected": len(lanes),
        "warning": None if len(lanes) == n_lanes else f"레인 {len(lanes)}개 검출 (예상 {n_lanes}개)",
    }


def _find_lane_centers(col_profile: np.ndarray, n_lanes: int) -> list:
    """
    열 강도 프로파일에서 n_lanes개의 레인 중심 열 인덱스를 찾습니다.
    피크 검출 실패 시 균등 분할로 폴백합니다.
    """
    w = len(col_profile)

    # 스무딩: 레인 너비의 약 1/3 크기로 이동평균
    smooth_w = max(3, w // (n_lanes * 3))
    kernel = np.ones(smooth_w) / smooth_w
    smoothed = np.convolve(col_profile, kernel, mode="same")

    # 로컬 최대값 찾기 (최소 거리: 레인 너비 × 0.6)
    min_dist = max(10, w // (n_lanes + 2))
    centers = _local_maxima(smoothed, min_dist=min_dist, n=n_lanes)

    if len(centers) >= n_lanes:
        centers = sorted(centers[:n_lanes])
        log.debug("레인 중심 피크 검출 성공: %s", centers)
        return centers

    # 폴백: 이미지 폭을 균등 분할
    log.info("레인 피크 검출 부족(%d개) → 균등 분할 폴백", len(centers))
    step = w / n_lanes
    return [int(step * i + step / 2) for i in range(n_lanes)]


def _local_maxima(arr: np.ndarray, min_dist: int, n: int) -> list:
    """배열에서 n개의 로컬 최대값 인덱스를 반환합니다 (greedy NMS)."""
    peaks = []
    remaining = arr.copy()
    for _ in range(n * 2):  # 충분한 후보 탐색
        idx = int(np.argmax(remaining))
        if remaining[idx] <= 0:
            break
        peaks.append(idx)
        # 이미 선택된 피크 주변 억제
        lo = max(0, idx - min_dist)
        hi = min(len(remaining), idx + min_dist + 1)
        remaining[lo:hi] = 0
    return peaks


def _extract_lane_features(lane_img: np.ndarray, label: str,
                            lane_index: int, global_max: float) -> dict:
    """단일 레인 슬라이스에서 밴드 특징을 추출합니다."""
    h, w = lane_img.shape
    base = {
        "lane_index": lane_index,
        "label": label,
        "log10_concentration": LOG10_CONC.get(label),
    }

    if h == 0 or w == 0:
        return {**base, "band_intensity": 0.0, "band_area": 0.0,
                "relative_intensity": 0.0, "band_width": 0.0, "band_height": 0.0,
                "is_saturated": False, "is_negative": True, "is_primer_dimer": False}

    # 행 방향 프로파일로 밴드 행 위치 추정
    row_profile = np.mean(lane_img, axis=1)
    band_row = int(np.argmax(row_profile))
    peak_intensity = float(row_profile[band_row])

    is_negative = peak_intensity < 8.0
    is_saturated = peak_intensity > 240.0

    if is_negative:
        return {**base, "band_intensity": 0.0, "band_area": 0.0,
                "relative_intensity": 0.0, "band_width": 0.0, "band_height": 0.0,
                "is_saturated": False, "is_negative": True, "is_primer_dimer": False}

    # 밴드 행 ±15% 범위에서 컨투어 탐색
    half_band = max(5, int(h * 0.15))
    row_start = max(0, band_row - half_band)
    row_end = min(h, band_row + half_band + 1)
    band_region = lane_img[row_start:row_end, :]

    # 임계값 이진화 (밴드 최대값의 30%)
    thresh_val = max(1, int(peak_intensity * 0.30))
    _, bin_mask = cv2.threshold(band_region, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 컨투어 없으면 band_region 전체 통계 사용
        band_intensity = float(np.mean(band_region[band_region > thresh_val])) \
            if np.any(band_region > thresh_val) else float(np.mean(band_region))
        band_area = float(np.sum(band_region > thresh_val))
        _, _, bw, bh = 0, 0, int(w), int(row_end - row_start)
    else:
        main_c = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(band_region)
        cv2.drawContours(mask, [main_c], -1, 255, thickness=cv2.FILLED)
        pixels = band_region[mask > 0]
        band_intensity = float(np.mean(pixels)) if pixels.size > 0 else 0.0
        band_area = float(cv2.contourArea(main_c))
        _, _, bw, bh = cv2.boundingRect(main_c)

    relative_intensity = band_intensity / global_max if global_max > 0 else 0.0

    # 프라이머 다이머 판별:
    # mecA 타겟 밴드(~533bp)는 젤 상단~중간에 위치하나,
    # 프라이머 다이머(<100bp)는 젤 하단 35% + 밴드 높이가 ROI의 8% 미만인 얇은 밴드로 나타남.
    band_position_ratio = band_row / max(h, 1)  # 0.0=상단, 1.0=하단
    is_primer_dimer = (
        band_position_ratio > 0.65
        and bh < h * 0.08
        and not is_saturated
    )

    return {
        **base,
        "band_intensity": round(band_intensity, 2),
        "band_area": round(band_area, 2),
        "relative_intensity": round(relative_intensity, 4),
        "band_width": float(bw),
        "band_height": float(bh),
        "is_saturated": bool(is_saturated),
        "is_negative": False,
        "is_primer_dimer": bool(is_primer_dimer),
    }


# ── 단일 밴드 추출 (하위 호환 유지) ─────────────────────────────────

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

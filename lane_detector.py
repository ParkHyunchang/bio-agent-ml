"""
PCR 젤 이미지에서 레인 위치(중심 열)를 검출하는 모듈.

주요 함수:
  detect_lanes()     → 이미지 전처리 + 레인 중심 검출, 이후 특징 추출에 필요한 컨텍스트 반환
  find_lane_centers() → 열 강도 프로파일에서 레인 중심 인덱스 검출
"""

import cv2
import numpy as np

from logger import get_logger

log = get_logger("lane_detector")


def detect_lanes(image_bytes: bytes, n_lanes: int = 10) -> dict:
    """
    이미지를 전처리하고 레인 중심 위치를 검출합니다.

    Returns:
        dict:
            image       (np.ndarray)  전처리된 그레이스케일 이미지 (블러 적용)
            lane_centers (list[int])  레인 중심 열 인덱스 리스트
            global_max  (float)       이미지 전체 최대 픽셀 강도
            roi_top     (int)         밴드 ROI 상단 행
            roi_bot     (int)         밴드 ROI 하단 행
            half_w      (int)         레인 절반 너비 (픽셀)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다. 지원 형식: JPEG, PNG, BMP, TIFF")

    h, w = img.shape

    # 방향 정규화: 밝은 배경 → 반전
    if float(np.mean(img)) > 128:
        img = cv2.bitwise_not(img)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    global_max = float(blurred.max()) or 1.0

    # 밴드 ROI: 상단 텍스트 10%, 하단 텍스트 8% 제외
    roi_top = int(h * 0.10)
    roi_bot = int(h * 0.92)
    roi = blurred[roi_top:roi_bot, :]

    col_profile = np.mean(roi, axis=0).astype(float)
    lane_centers = find_lane_centers(col_profile, n_lanes)

    half_w = max(w // (n_lanes * 2), 20)

    return {
        "image": blurred,
        "lane_centers": lane_centers,
        "global_max": global_max,
        "roi_top": roi_top,
        "roi_bot": roi_bot,
        "half_w": half_w,
    }


def find_lane_centers(col_profile: np.ndarray, n_lanes: int) -> list:
    """
    열 강도 프로파일에서 n_lanes개의 레인 중심 열 인덱스를 반환합니다.
    피크 검출 실패 시 균등 분할로 폴백합니다.
    """
    w = len(col_profile)

    smooth_w = max(3, w // (n_lanes * 3))
    kernel = np.ones(smooth_w) / smooth_w
    smoothed = np.convolve(col_profile, kernel, mode="same")

    min_dist = max(10, w // (n_lanes + 2))
    centers = _local_maxima(smoothed, min_dist=min_dist, n=n_lanes)

    if len(centers) >= n_lanes:
        centers = sorted(centers[:n_lanes])
        log.debug("레인 중심 피크 검출 성공: %s", centers)
        return centers

    log.info("레인 피크 검출 부족(%d개) → 균등 분할 폴백", len(centers))
    step = w / n_lanes
    return [int(step * i + step / 2) for i in range(n_lanes)]


def _local_maxima(arr: np.ndarray, min_dist: int, n: int) -> list:
    """배열에서 n개의 로컬 최대값 인덱스를 반환합니다 (greedy NMS)."""
    peaks = []
    remaining = arr.copy()
    for _ in range(n * 2):
        idx = int(np.argmax(remaining))
        if remaining[idx] <= 0:
            break
        peaks.append(idx)
        lo = max(0, idx - min_dist)
        hi = min(len(remaining), idx + min_dist + 1)
        remaining[lo:hi] = 0
    return peaks

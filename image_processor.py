"""
PCR 젤 전기영동 이미지에서 밴드 특징을 추출하는 모듈.

처리 파이프라인:
  1. 그레이스케일 변환
  2. 이미지 방향 정규화 (밝은 밴드 기준)
  3. 가우시안 블러로 노이즈 제거
  4. OTSU 임계값으로 밴드 영역 이진화
  5. 컨투어 기반 밴드 검출
  6. 특징값 추출: band_intensity, band_area, relative_intensity, band_width
"""

import cv2
import numpy as np


def process_gel_image(image_bytes: bytes) -> dict:
    """
    PCR 젤 이미지 바이트에서 밴드 특징을 추출합니다.

    Args:
        image_bytes: 이미지 파일의 raw 바이트 (JPEG/PNG/BMP 등)

    Returns:
        dict:
            band_intensity      (float) 밴드 영역 평균 픽셀 밝기 (0–255)
            band_area           (float) 밴드 픽셀 면적 (px²)
            relative_intensity  (float) 이미지 최대 밝기 대비 상대값 (0–1)
            band_width          (float) 밴드 수평 너비 (px)
            lanes_detected      (int)   검출된 밴드(컨투어) 수
            warning             (str)   경고 메시지 (밴드 검출 실패 시)

    Raises:
        ValueError: 이미지 디코딩 실패 시
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다. 지원 형식: JPEG, PNG, BMP, TIFF")

    # ── 1. 방향 정규화 ────────────────────────────────────────────────
    # EtBr/SYBR 형광 이미지: 어두운 배경 + 밝은 밴드 (반전 불필요)
    # 투과광 이미지: 밝은 배경 + 어두운 밴드 (반전 필요)
    mean_brightness = float(np.mean(img))
    if mean_brightness > 128:
        # 밝은 배경 → 반전하여 밴드를 밝게 만듦
        img = cv2.bitwise_not(img)

    # ── 2. 노이즈 제거 ────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # ── 3. 이진화 (OTSU) ─────────────────────────────────────────────
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── 4. 컨투어 검출 ────────────────────────────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # OTSU 실패 시 평균+표준편차 기반 임계값으로 재시도
    if not contours:
        manual_thresh = mean_brightness + float(np.std(blurred))
        _, thresh2 = cv2.threshold(blurred, int(manual_thresh), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── 5. 밴드 검출 실패 폴백 ───────────────────────────────────────
    if not contours:
        return {
            "band_intensity": float(np.mean(blurred)),
            "band_area": float(blurred.size),
            "relative_intensity": float(np.mean(blurred)) / 255.0,
            "band_width": float(blurred.shape[1]),
            "lanes_detected": 0,
            "warning": "밴드 자동 검출 실패, 전체 이미지 통계를 사용합니다.",
        }

    # ── 6. 메인 밴드 선택 (면적 기준 최대 컨투어) ───────────────────
    main_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(blurred)
    cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

    band_pixels = blurred[mask > 0]
    band_intensity = float(np.mean(band_pixels)) if band_pixels.size > 0 else 0.0
    band_area = float(cv2.contourArea(main_contour))
    _, _, band_width, _ = cv2.boundingRect(main_contour)

    global_max = float(blurred.max())
    relative_intensity = band_intensity / global_max if global_max > 0 else 0.0

    result = {
        "band_intensity": band_intensity,
        "band_area": band_area,
        "relative_intensity": relative_intensity,
        "band_width": float(band_width),
        "lanes_detected": len(contours),
    }

    # 경고: 단일 밴드만 검출된 경우 래더 정규화 불가
    if len(contours) == 1:
        result["warning"] = "밴드 1개만 검출됨. 래더 포함 여부를 확인하세요."

    return result

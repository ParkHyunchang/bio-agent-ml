"""
로깅 설정.
  LOG_TO_FILE=true → logs/app.log 파일 출력 (일별 롤링, 30일 보관)
  그 외 (기본값)   → stdout 출력 (Docker logs 캡처 가능)
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler

from dotenv import load_dotenv

load_dotenv()  # .env 파일이 있으면 자동 로드 (없어도 무시)

# 컨테이너 OS 시계가 UTC여도 Python의 localtime을 KST로 강제.
# → TimedRotatingFileHandler 회전이 KST 자정에 일어나고, 파일 suffix도 KST 날짜로 기록됨.
# POSIX TZ 문자열("KST-9")은 zoneinfo 파일 없이 동작 → tzdata 패키지 불필요.
# Windows 등 tzset 미지원 환경에서는 OS 시계를 그대로 사용.
os.environ.setdefault("TZ", "KST-9")
if hasattr(time, "tzset"):
    time.tzset()

_LOG_TO_FILE = os.getenv("LOG_TO_FILE", "").lower() == "true"

_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 컨테이너 시계가 UTC여도 로그는 항상 KST로 출력 (운영 환경 일관성)
_KST = timezone(timedelta(hours=9))


class KSTFormatter(logging.Formatter):
    """asctime을 KST 기준으로 변환하는 Formatter."""

    def converter(self, timestamp):  # type: ignore[override]
        return datetime.fromtimestamp(timestamp, tz=_KST).timetuple()


def setup_logging() -> None:
    """루트 로거 + uvicorn 로거를 일괄 설정합니다. 앱 시작 시 1회 호출."""
    handler = _build_handler()
    handler.setFormatter(KSTFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    # uvicorn 로거가 루트로 전파되도록 설정 (중복 출력 방지)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 반환합니다."""
    return logging.getLogger(name)


def _build_handler() -> logging.Handler:
    if _LOG_TO_FILE:
        os.makedirs("logs", exist_ok=True)
        handler = TimedRotatingFileHandler(
            filename="logs/app.log",
            when="midnight",
            backupCount=30,
            encoding="utf-8",
        )
        handler.suffix = "%Y-%m-%d"
    else:
        handler = logging.StreamHandler(sys.stdout)
    return handler

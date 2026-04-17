"""
환경에 따른 로깅 설정.
  APP_ENV=local (기본값) → 터미널 출력
  APP_ENV=production     → logs/app.log 파일 출력 (일별 롤링, 30일 보관)
"""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from dotenv import load_dotenv

load_dotenv()  # .env 파일이 있으면 자동 로드 (없어도 무시)

APP_ENV = os.getenv("APP_ENV", "local")

_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """루트 로거 + uvicorn 로거를 일괄 설정합니다. 앱 시작 시 1회 호출."""
    handler = _build_handler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

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
    if APP_ENV != "local":
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

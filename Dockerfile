FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 (OpenCV headless 의존성)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 모델 저장 디렉토리 생성
RUN mkdir -p model

ENV APP_ENV=production

EXPOSE 3212

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3212"]

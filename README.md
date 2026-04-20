# bio-agent-ml

PCR 젤 전기영동 이미지에서 qPCR Ct값을 예측하는 회귀 모델 마이크로서비스.

## 기술 스택

- **FastAPI** — REST API 서버 (port 3212)
- **OpenCV** — 젤 이미지 밴드 특징 추출
- **scikit-learn** — 회귀 모델 학습 및 예측

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/extract` | 이미지 → 밴드 특징 JSON 반환 |
| POST | `/train` | 훈련 데이터로 모델 학습 |
| POST | `/predict` | 새 이미지 → Ct값 예측 |
| GET | `/model/status` | 모델 상태 조회 |
| GET | `/health` | 헬스체크 |

## 로컬 실행
```bash
python --version
python -m pip install -r requirements.txt
python main.py
```

```bash
pip install -r requirements.txt
python main.py
# → http://localhost:3212/docs (Swagger UI)
```

```bash
netstat -ano | findstr :3212
taskkill /F /PID [확인된PID]
or
 Get-Process -Id [확인된PID]
Stop-Process -Id [확인된PID] -Force
```

## Docker 실행

```bash
docker build -t bio-agent-ml .
docker run -p 3212:3212 -v $(pwd)/model:/app/model bio-agent-ml
```

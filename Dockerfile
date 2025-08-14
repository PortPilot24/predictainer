# ----- Base -----
FROM python:3.10-slim

# 파이썬/스레드/로그 기본 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# 런타임에 필요한 OS 패키지 (LightGBM/CatBoost용 OpenMP)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ----- App -----
WORKDIR /app

# 의존성 먼저 설치(레이어 캐시 극대화)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 앱 소스/모델 복사
COPY . .

# 비루트 사용자로 실행 + 권한 정리
RUN useradd -m -s /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

# (선택) 문서화용
EXPOSE 8000

# ⚠ App Service가 자체 프로브로 /health를 확인하므로 Dockerfile HEALTHCHECK는 제거
# HEALTHCHECK ...  <- 사용 안 함

# ----- Run -----
# App Service는 WEBSITES_PORT를 주입함. 없으면 8000 사용.
# lifespan=on: FastAPI lifespan 훅(백그라운드 모델 로드) 강제 활성화
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${WEBSITES_PORT:-8000} --workers 1 --lifespan on --log-level info"]
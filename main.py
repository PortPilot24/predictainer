# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import asyncio
import logging
import os
import pickle
import requests
import holidays
import numpy as np

# (로컬 개발 편의를 위해) .env 지원 — 프로덕션(App Service)은 앱 설정으로 주입됨
try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

# ========= FastAPI 기본 설정 =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="predictainer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요시 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= 전역 상태 =========
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logging.warning("API_KEY env not set. Set it in App Service > Configuration > Application settings.")

models_ready: bool = False
load_error: str | None = None

# 모델/스케일러 전역 핸들 (로드 후 할당)
lgb_model = None
cat_model = None
mlp_model = None
scaler = None


# ========= 모델 로더 =========
def _blocking_load_models():
    """블로킹 모델 로드: 별도 스레드에서 실행"""
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    import tensorflow as tf

    global lgb_model, cat_model, mlp_model, scaler

    # 경로는 컨테이너 /app 기준
    lgb_model = lgb.Booster(model_file="models/lgb_model.txt")

    cat = CatBoostRegressor()
    cat.load_model("models/cat_model.cbm")
    cat_model = cat

    mlp_model = tf.keras.models.load_model("models/mlp_model.h5", compile=False)

    with open("models/scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)
    globals()["scaler"] = _scaler


async def load_models_async():
    """이벤트 루프를 막지 않도록, 블로킹 로드를 워커 스레드로 넘김"""
    start = datetime.utcnow()
    logging.info("🔄 Loading models in background...")
    await asyncio.to_thread(_blocking_load_models)
    took = (datetime.utcnow() - start).total_seconds()
    logging.info(f"✅ Models loaded (took {took:.1f}s)")


# ========= 앱 라이프사이클: 서버가 포트를 연 뒤 백그라운드로 로드 =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_ready, load_error
    try:
        # 백그라운드에서 모델 로드 시작
        async def _runner():
            global models_ready, load_error
            try:
                await load_models_async()
                models_ready = True
                load_error = None
            except Exception as e:
                logging.exception("Model loading failed")
                models_ready = False
                load_error = str(e)

        asyncio.create_task(_runner())
    except Exception as e:
        logging.exception("Failed to schedule model loading")
        models_ready = False
        load_error = str(e)
    yield
    # 종료 시 정리 필요하면 여기서


app.router.lifespan_context = lifespan


# ========= 유틸 =========
def _build_features(daily: dict, date_obj: datetime) -> list[float]:
    avg_temp = float(daily["temp"]["day"])
    min_temp = float(daily["temp"]["min"])
    max_temp = float(daily["temp"]["max"])
    wind_speed = float(daily.get("wind_speed", 0.0))
    humidity = float(daily.get("humidity", 0.0))
    precipitation = float(daily.get("rain", 0.0))  # 없으면 0.0

    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    weekday = date_obj.weekday()
    dayofyear = date_obj.timetuple().tm_yday
    weekofyear = int(date_obj.strftime("%U"))
    is_end_of_month = 1 if (date_obj + timedelta(days=1)).month != month else 0

    kr_holidays = holidays.KR(years=[year])
    is_holiday = 1 if date_obj.date() in kr_holidays else 0
    is_monday_after_holiday = 1 if (weekday == 0 and (date_obj - timedelta(days=1)).date() in kr_holidays) else 0
    season = 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8] else 3 if month in [9, 10, 11] else 4

    return [
        avg_temp, min_temp, max_temp, precipitation, wind_speed, humidity,
        year, month, day, weekday, dayofyear, weekofyear,
        is_holiday, is_end_of_month, is_monday_after_holiday, season
    ]


# ========= 라우트 =========
@app.get("/health", tags=["health"])
def health():
    """App Service / AFD 헬스 프로브용 — 항상 가볍게 200"""
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
def ready():
    """모델 로드 준비 여부 확인용(선택)"""
    return {"ready": models_ready, "error": load_error}


@app.get("/predictainer/predict")
def predict_8days():
    # 모델 준비 안 됐으면 503
    if not models_ready:
        raise HTTPException(status_code=503, detail="Model is warming up. Try again in a moment.")

    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY is not configured on the server.")

    # 1) 날씨 호출
    try:
        url = "https://api.openweathermap.org/data/3.0/onecall"
        params = {
            "lat": 34.901993,
            "lon": 127.659044,
            "units": "metric",
            "lang": "kr",
            "exclude": "current,minutely,hourly,alerts",
            "appid": API_KEY,
        }
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        weather_data = resp.json()
    except requests.RequestException as e:
        logging.warning(f"OpenWeather request failed: {e}")
        raise HTTPException(status_code=502, detail="Weather API request failed")

    daily_list = weather_data.get("daily", [])
    if not daily_list:
        raise HTTPException(status_code=502, detail="Weather API returned no daily data")

    # 2) 예측
    preds = []
    for daily in daily_list:
        date_obj = datetime.utcfromtimestamp(daily["dt"]) + timedelta(hours=9)

        features = _build_features(daily, date_obj)
        features_np = np.array([features], dtype=float)

        # 스케일링은 MLP에만 적용(원 코드 유지)
        scaled = scaler.transform(features_np)

        try:
            # LightGBM
            pred_lgb = float(lgb_model.predict(features_np)[0])
            # CatBoost
            pred_cat = float(cat_model.predict(features_np)[0])
            # TensorFlow
            pred_mlp = float(mlp_model.predict(scaled, verbose=0)[0][0])
        except Exception as e:
            logging.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

        ensemble = 0.6 * pred_lgb + 0.1 * pred_cat + 0.3 * pred_mlp

        preds.append({
            "date": date_obj.strftime("%Y-%m-%d"),
            "weather": {
                "avg_temp": daily["temp"]["day"],
                "min_temp": daily["temp"]["min"],
                "max_temp": daily["temp"]["max"],
                "precipitation": daily.get("rain", 0.0),
                "wind_speed": daily.get("wind_speed", 0.0),
                "humidity": daily.get("humidity", 0.0),
            },
            "prediction": {"Ensemble": int(ensemble)}
        })

    return {"predictions": preds}

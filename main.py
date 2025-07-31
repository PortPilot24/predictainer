from fastapi import FastAPI, Query
from datetime import datetime, timedelta
import requests
import holidays
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

app = FastAPI()

# ✅ 환경 변수 로드 (.env에 API_KEY가 저장되어 있다고 가정)
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ✅ 모델 & 스케일러 불러오기
lgb_model = lgb.Booster(model_file="models/lgb_model.txt")

cat_model = CatBoostRegressor()
cat_model.load_model("models/cat_model.cbm")

mlp_model = tf.keras.models.load_model("models/mlp_model.h5")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ 8일치 예측 API
@app.get("/predict/8days")
def predict_8days():
    # 1. OpenWeatherMap API 호출
    url = f"https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": 34.901993,
        "lon": 127.659044,
        "units": "metric",
        "lang": "kr",
        "exclude": "current,minutely,hourly,alerts",
        "appid": API_KEY
    }
    response = requests.get(url, params=params)
    weather_data = response.json()

    predictions = []

    for daily in weather_data.get("daily", []):
        # 2. 날짜 정보
        date_obj = datetime.utcfromtimestamp(daily["dt"]) + timedelta(hours=9)

        # 3. 날씨 정보
        avg_temp = daily["temp"]["day"]
        min_temp = daily["temp"]["min"]
        max_temp = daily["temp"]["max"]
        wind_speed = daily["wind_speed"]
        humidity = daily["humidity"]
        precipitation = daily.get("rain", 0.0)  # 없을 경우 0.0

        # 4. 파생 변수
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        weekday = date_obj.weekday()
        dayofyear = date_obj.timetuple().tm_yday
        weekofyear = int(date_obj.strftime("%U"))
        is_weekend = 1 if weekday in [5, 6] else 0
        is_start_of_month = 1 if day in [1, 2, 3] else 0
        is_end_of_month = 1 if (date_obj + timedelta(days=1)).month != month else 0
        is_quarter_end = 1 if month in [3, 6, 9, 12] and is_end_of_month else 0

        kr_holidays = holidays.KR(years=[year])
        is_holiday = 1 if date_obj.date() in kr_holidays else 0
        is_day_off = 1 if is_weekend or is_holiday else 0
        is_monday_after_holiday = 1 if (weekday == 0 and (date_obj - timedelta(days=1)).date() in kr_holidays) else 0
        season = 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8] else 3 if month in [9, 10, 11] else 4

        input_features = [
            avg_temp, min_temp, max_temp, precipitation, wind_speed, humidity,
            year, month, day, weekday, dayofyear, weekofyear,
            is_weekend, is_holiday, is_day_off,
            is_start_of_month, is_end_of_month, is_quarter_end,
            is_monday_after_holiday, season
        ]

        input_scaled = scaler.transform([input_features])
        pred_lgb = lgb_model.predict([input_features])[0]
        pred_cat = cat_model.predict([input_features])[0]
        pred_mlp = mlp_model.predict(input_scaled)[0][0]
        ensemble = 0.4 * pred_lgb + 0.2 * pred_cat + 0.4 * pred_mlp

        predictions.append({
            "date": date_obj.strftime("%Y-%m-%d"),
            "weather": {
                "avg_temp": avg_temp,
                "min_temp": min_temp,
                "max_temp": max_temp,
                "precipitation": precipitation,
                "wind_speed": wind_speed,
                "humidity": humidity
            },
            "prediction": {
                "LightGBM": int(pred_lgb),
                "CatBoost": int(pred_cat),
                "MLP": int(pred_mlp),
                "Ensemble": int(ensemble)
            }
        })

    return {"predictions": predictions}

# scripts/check_predictions.py
import requests, json, time
base = "http://127.0.0.1:8000"
r = requests.post(base+"/api/predict", json={"timestamp":"2025-12-05T05:30:00","ph":8.2,"tds":420})
print("PREDICT:", r.json())
rows = [{"timestamp":"2025-12-05T05:00:00","ph":8.0,"tds":410},{"timestamp":"2025-12-05T05:05:00","ph":8.1,"tds":412}]
r2 = requests.post(base+"/api/forecast_lstm", json=rows)
print("LSTM FORECAST:", r2.json())

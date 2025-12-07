# Dashboard README

This document explains how to configure and use the Water Intelligence Dashboard.

## Overview

The dashboard (`index.html`) is a single-page application that communicates with the FastAPI backend to:

1. Submit water quality readings (pH, TDS)
2. Get real-time quality predictions and safety recommendations
3. View forecasted trends for pH and TDS
4. See active alerts and model status

## Getting Started

1. Start the backend server:

   ```bash
   cd src/backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Open http://localhost:8000 in your browser

3. The dashboard will load automatically from the static files

## API Endpoints

### POST /api/predict

Submit readings for quality prediction.

**Request:**

```json
{
  "readings": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "ph": 7.2,
      "tds": 350
    }
  ]
}
```

**Response:**

```json
{
  "water_quality_index": 82.5,
  "ph_virtual": 7.15,
  "tds_virtual": 345.2,
  "forecast_ph": [7.1, 7.0, 6.95, 6.9, 6.85],
  "forecast_tds": [350, 360, 375, 390, 410],
  "is_anomaly": false,
  "label": "SAFE",
  "recommendation": {
    "status": "SAFE",
    "summary": "Water quality is within acceptable limits",
    "actions": ["Continue monitoring at regular intervals"]
  },
  "alerts": []
}
```

### POST /api/forecast

Get time-series forecasts.

**Request:**

```json
{
  "readings": [
    { "timestamp": "2024-01-15T10:00:00", "ph": 7.0, "tds": 300 },
    { "timestamp": "2024-01-15T10:05:00", "ph": 7.1, "tds": 310 },
    { "timestamp": "2024-01-15T10:10:00", "ph": 7.2, "tds": 320 }
  ],
  "steps": 10
}
```

**Response:**

```json
{
  "ph": [7.3, 7.35, 7.4, 7.42, 7.45, 7.47, 7.5, 7.52, 7.55, 7.57],
  "tds": [330, 340, 350, 358, 365, 372, 380, 387, 395, 402]
}
```

### GET /api/health

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

### GET /api/models

List loaded models.

**Response:**

```json
{
  "models": [
    "rf_virtual_ph",
    "rf_virtual_tds",
    "rf_forecast_ph",
    "rf_forecast_tds",
    "iforest_anomaly",
    "rf_classifier",
    "lstm_ph",
    "lstm_tds"
  ]
}
```

## Data Validation

All input is validated using Pydantic models:

| Field       | Type     | Constraints                   |
| ----------- | -------- | ----------------------------- |
| `timestamp` | datetime | ISO 8601 format               |
| `ph`        | float    | 0.0 ≤ ph ≤ 14.0               |
| `tds`       | float    | tds ≥ 0                       |
| `steps`     | int      | 1 ≤ steps ≤ 100 (default: 10) |

## UI Components

### Input Panel

- pH input: Numeric field with 0-14 range
- TDS input: Numeric field with 0+ range
- Submit button: Sends reading to /api/predict

### Prediction Card

- Water Quality Index: 0-100 score
- Status badge: SAFE (green), STRESS (yellow), DANGER (red)
- Virtual sensor readings: Cleaned pH and TDS values

### Forecast Chart

- Interactive Chart.js line chart
- Shows forecasted pH (blue) and TDS (orange)
- Hover for exact values

### Alerts Panel

- Lists any anomalies detected
- Shows threshold violations
- Color-coded by severity

### Model Status

- Lists all loaded models
- Shows load status (✓ loaded, ✗ failed)
- Click to expand details

## Customization

### Themes

The dashboard uses Tailwind CSS via CDN. To customize:

1. Edit the `<script>` tag with Tailwind config
2. Modify class names for colors
3. Update the CSS variables if needed

### Adding Endpoints

To add new API integration:

```javascript
async function callNewEndpoint(data) {
  const response = await fetch('/api/new-endpoint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  return await response.json()
}
```

## Troubleshooting

### CORS Errors

If you see CORS errors in the console:

1. Check that `CORS_ORIGINS` in `.env` includes your domain
2. Restart the server after changing `.env`

### Models Not Loading

If models show as "not loaded":

1. Ensure model files exist in `models/` directory
2. Check for `model_card.json` files
3. Review server logs for load errors

### Chart Not Displaying

If the forecast chart is blank:

1. Check browser console for JavaScript errors
2. Ensure forecast data has valid arrays
3. Try refreshing the page

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run all tests
pytest tests/

# Run API tests only
pytest tests/test_api_health.py -v
```

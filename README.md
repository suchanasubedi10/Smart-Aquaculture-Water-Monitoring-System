# 🌊 Water Intelligence Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e?logo=scikit-learn&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.0+-38bdf8?logo=tailwind-css&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-AI_Agent-9333ea?logo=openai&logoColor=white)

**An AI-Powered Aquaculture Water Quality Monitoring & Prediction System**

_Real-time monitoring, ML-based predictions, LSTM time-series forecasting, and intelligent recommendations for optimal water quality management_

[Features](#-key-features) • [Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Dashboard](#-dashboard) • [Models](#-machine-learning-models)

</div>

---

## 📖 Project Overview

The **Water Intelligence Project** is a comprehensive, production-ready system designed for **aquaculture and fish farming operations** to monitor, predict, and optimize water quality parameters. It combines classical machine learning with deep learning techniques to provide:

- **Real-time water quality classification** (SAFE / STRESS / DANGER)
- **Time-series forecasting** of pH and TDS using LSTM neural networks
- **Anomaly detection** for early warning of water quality issues
- **Virtual sensors** to estimate missing parameters
- **AI-powered chatbot** (AquaBot) for intelligent recommendations
- **Interactive dashboard** with comprehensive visualizations

### 🎯 Problem Statement

Water quality is critical in aquaculture - poor conditions lead to fish stress, disease, and mortality. Traditional monitoring is:

- **Reactive** rather than predictive
- **Manual** and time-consuming
- **Limited** in providing actionable insights

### 💡 Our Solution

This project transforms water quality management from reactive to **proactive** by:

1. **Predicting future values** before problems occur
2. **Classifying conditions** instantly with confidence scores
3. **Detecting anomalies** automatically
4. **Recommending actions** through AI assistance

---

## ✨ Key Features

### 🔬 Machine Learning Pipeline

| Feature                     | Technology               | Description                                                        |
| --------------------------- | ------------------------ | ------------------------------------------------------------------ |
| **Quality Classification**  | Random Forest            | 3-class classification (SAFE/STRESS/DANGER) with confidence scores |
| **Time-Series Forecasting** | LSTM (PyTorch)           | Multi-step pH and TDS predictions with sequence modeling           |
| **Anomaly Detection**       | Isolation Forest         | Unsupervised outlier detection for early warning                   |
| **Virtual Sensors**         | Random Forest Regressors | Estimate DO, Temperature, Turbidity from available data            |
| **RF Forecasting**          | Random Forest            | Alternative forecasting using feature engineering                  |

### 🖥️ Interactive Dashboard

| Component                  | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| **Real-Time Predictions**  | Instant classification with trend indicators and confidence    |
| **LSTM Forecast Charts**   | Interactive time-series visualization with fullscreen mode     |
| **Water Quality Gauge**    | Circular gauge showing overall water quality score             |
| **Anomaly Timeline**       | Scatter plot of detected anomalies over time                   |
| **Risk Heatmap**           | 24×7 grid showing risk patterns by hour and day                |
| **Sensor Health Monitor**  | Real-time sensor status, drift detection, calibration tracking |
| **AI Parameter Simulator** | What-if scenario testing for water parameters                  |
| **Event Timeline**         | Chronological log of predictions and events                    |
| **PDF Report Generation**  | Downloadable comprehensive analysis reports                    |

### 🤖 AI Assistant (AquaBot)

Powered by **LangGraph + Google Generative AI** with conversation persistence:

- Natural language questions about water quality
- Context-aware recommendations based on current predictions
- Voice input support (Web Speech API)
- Multi-turn conversations with memory

### ⚡ Additional Features

- **Dark Mode** - Full theme support across dashboard
- **Keyboard Shortcuts** - Quick actions (Ctrl+P, Ctrl+F, Ctrl+D, etc.)
- **Quick Scenarios** - Pre-configured SAFE/STRESS/DANGER test cases
- **Live Statistics** - Session averages, ranges, and anomaly rates
- **Connection Status** - Real-time API health with latency monitoring
- **Data Export** - CSV download of prediction history
- **Responsive Design** - Works on desktop and mobile devices

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WATER INTELLIGENCE SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   SENSORS    │    │              DASHBOARD (Frontend)                 │   │
│  │              │    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │   │
│  │  • pH Sensor │───▶│  │Predict  │ │Forecast │ │Sensors  │ │Assistant│ │   │
│  │  • TDS Sensor│    │  │Tab      │ │Tab      │ │Tab      │ │(AquaBot)│ │   │
│  │  • Temp      │    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ │   │
│  │  • DO        │    │       │           │           │           │       │   │
│  └──────────────┘    └───────┼───────────┼───────────┼───────────┼───────┘   │
│                              │           │           │           │           │
│                              ▼           ▼           ▼           ▼           │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI BACKEND (REST API)                      │   │
│  │                                                                         │   │
│  │  /api/predict    /api/forecast    /api/forecast_lstm    /api/chat     │   │
│  │  /api/virtual    /api/models      /api/sensor_health    /api/report   │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                              │           │           │           │           │
│                              ▼           ▼           ▼           ▼           │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                        ML/DL MODEL LAYER                               │   │
│  │                                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │
│  │  │ Classifier  │  │   LSTM      │  │  Anomaly    │  │   Virtual   │   │   │
│  │  │(RandomForest│  │ Forecasters │  │  Detector   │  │   Sensors   │   │   │
│  │  │ 3-class)    │  │ (PyTorch)   │  │(IsolationF) │  │    (RF)     │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 🐳 Option 1: Docker (Recommended - Easiest)

Just install Docker and run:

```bash
# Clone the repository
git clone <repository-url>
cd water-intel-project

# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t water-intel .
docker run -p 8000:8000 water-intel
```

**That's it!** Open http://localhost:8000/static/index.html in your browser.

> **Note**: The Docker image includes pre-trained models and sample data. For the AI Assistant (AquaBot), set your Google API key:
>
> ```bash
> docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key_here water-intel
> ```

### 🐍 Option 2: Manual Installation

#### Prerequisites

- **Python 3.10+**
- **pip** package manager
- **Google API Key** (for AI Assistant - optional)

#### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd water-intel-project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional - for AI Assistant)
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Data Generation & Model Training

```bash
# Generate synthetic training data (10,000 samples)
python scripts/gen_synthetic_full.py --out-dir data --n-samples 10000 --seed 42

# Train classical ML models (Classifier, Anomaly Detector, Virtual Sensors)
python scripts/train_all_models.py --data-dir data --models-dir models --seed 42

# Train LSTM forecasters (pH and TDS)
python scripts/train_lstm_pytorch.py --data-csv data/cleaned.csv --models-dir models --epochs 50 --seed 42
```

### Launch the Application

```bash
# Start the API server
cd src/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access the Dashboard

| URL                                     | Description                 |
| --------------------------------------- | --------------------------- |
| http://127.0.0.1:8000/static/index.html | **Main Dashboard**          |
| http://127.0.0.1:8000/docs              | API Documentation (Swagger) |
| http://127.0.0.1:8000/redoc             | API Documentation (ReDoc)   |

---

## 📊 API Reference

### Core Endpoints

| Method | Endpoint             | Description                                                                       |
| ------ | -------------------- | --------------------------------------------------------------------------------- |
| `GET`  | `/api/health`        | API health check with status                                                      |
| `GET`  | `/api/models`        | List all loaded ML models                                                         |
| `POST` | `/api/predict`       | **Main prediction endpoint** - classification, anomaly detection, recommendations |
| `POST` | `/api/forecast`      | Random Forest time-series forecast                                                |
| `POST` | `/api/forecast_lstm` | **LSTM neural network forecast**                                                  |
| `POST` | `/api/virtual`       | Virtual sensor predictions                                                        |
| `POST` | `/api/feature_row`   | Compute derived features                                                          |

### AI & Reporting Endpoints

| Method | Endpoint               | Description                           |
| ------ | ---------------------- | ------------------------------------- |
| `POST` | `/api/chat`            | Conversational AI assistant (AquaBot) |
| `POST` | `/api/report/generate` | Generate PDF analysis report          |
| `GET`  | `/api/sensor_health`   | Sensor status and calibration         |

### Example: Prediction Request

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-01T12:00:00Z",
    "ph": 7.2,
    "tds": 350,
    "device_id": "sensor_1"
  }'
```

### Example: Response

```json
{
  "label": "SAFE",
  "label_id": 0,
  "confidence": 0.92,
  "is_anomaly": false,
  "anomaly_score": 0.15,
  "virtual_sensors": {
    "do_pred": 7.1,
    "temp_pred": 26.5,
    "turb_pred": 18.2
  },
  "recommendations": [
    {
      "type": "info",
      "parameter": "general",
      "message": "All parameters within optimal range",
      "action": "Continue routine monitoring"
    }
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## 🖥️ Dashboard

### Navigation Tabs

| Tab              | Description                                             |
| ---------------- | ------------------------------------------------------- |
| **📊 Predict**   | Main prediction interface with real-time classification |
| **📈 Forecast**  | LSTM time-series forecasting with interactive charts    |
| **🔧 Sensors**   | Sensor health monitoring and drift detection            |
| **🤖 Assistant** | AI-powered chatbot for water quality guidance           |
| **📋 Events**    | Timeline of predictions and system events               |
| **📄 Reports**   | PDF report generation and downloads                     |

### Keyboard Shortcuts

| Shortcut | Action               |
| -------- | -------------------- |
| `1-6`    | Switch between tabs  |
| `Ctrl+P` | Run prediction       |
| `Ctrl+F` | Run forecast         |
| `Ctrl+D` | Toggle dark mode     |
| `Ctrl+E` | Export CSV           |
| `Ctrl+/` | Show shortcuts help  |
| `S`      | Load SAFE scenario   |
| `T`      | Load STRESS scenario |
| `X`      | Load DANGER scenario |

### Quick Scenario Testing

Pre-configured test values calibrated to the trained model:

| Scenario      | pH  | TDS | Expected Result       |
| ------------- | --- | --- | --------------------- |
| 🟢 **SAFE**   | 7.0 | 300 | Optimal conditions    |
| 🟡 **STRESS** | 6.3 | 400 | Borderline conditions |
| 🔴 **DANGER** | 5.9 | 200 | Critical alert        |

---

## 🤖 Machine Learning Models

### Model Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CLASSIFICATION (3-Class)                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Input: [ph, tds, hour, ph_rolling, tds_rolling, ...]   │    │
│  │           ↓                                              │    │
│  │  Random Forest Classifier (100 trees)                    │    │
│  │           ↓                                              │    │
│  │  Output: SAFE (0) | STRESS (1) | DANGER (2)             │    │
│  │          + Confidence Probability                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  LSTM FORECASTING (Sequence-to-Value)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Input: Sequence of [timestamp, ph, tds] (window=10)    │    │
│  │           ↓                                              │    │
│  │  LSTM Layer 1 (hidden=64, dropout=0.2)                  │    │
│  │           ↓                                              │    │
│  │  LSTM Layer 2 (hidden=64, dropout=0.2)                  │    │
│  │           ↓                                              │    │
│  │  Linear Layer → Output: Next pH/TDS value               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ANOMALY DETECTION (Unsupervised)                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Input: Feature vector                                   │    │
│  │           ↓                                              │    │
│  │  Isolation Forest (contamination=0.05)                  │    │
│  │           ↓                                              │    │
│  │  Output: Normal (+1) | Anomaly (-1) + Score             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  VIRTUAL SENSORS (Regression)                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Input: [ph, tds, hour_sin, hour_cos]                   │    │
│  │           ↓                                              │    │
│  │  Random Forest Regressor (per sensor)                   │    │
│  │           ↓                                              │    │
│  │  Output: Estimated DO, Temperature, Turbidity           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Model Performance Metrics

| Model      | Metric    | Value |
| ---------- | --------- | ----- |
| Classifier | Accuracy  | ~95%  |
| Classifier | F1-Score  | ~0.94 |
| LSTM pH    | MAE       | ~0.08 |
| LSTM TDS   | MAE       | ~12.5 |
| Anomaly    | Precision | ~0.90 |

### Model Cards

Each trained model includes a comprehensive model card (`*.model_card.json`) containing:

- Training date and hyperparameters
- Performance metrics
- Input/output schemas
- Version information

---

## 📁 Project Structure

```
water-intel-project/
│
├── 📂 data/                          # Data directory
│   ├── raw_combined.csv              # Generated raw sensor data
│   ├── cleaned.csv                   # Preprocessed data
│   ├── features_labeled.csv          # Feature-engineered dataset
│   ├── sample_summary.json           # Data statistics
│   ├── reports/                      # Generated PDF reports
│   └── checkpoints/                  # LangGraph conversation memory
│
├── 📂 models/                        # Trained models
│   ├── classifier_rf.model_card.json # Classifier metadata
│   ├── anomaly_if.model_card.json    # Anomaly detector metadata
│   ├── forecast_ph_lstm_pt.pth       # LSTM pH model weights
│   ├── forecast_tds_lstm_pt.pth      # LSTM TDS model weights
│   ├── virtual_*.model_card.json     # Virtual sensor models
│   ├── feature_list.json             # Feature configuration
│   └── metrics_*.json                # Training metrics
│
├── 📂 scripts/                       # Training & utility scripts
│   ├── gen_synthetic_full.py         # Synthetic data generation
│   ├── train_all_models.py           # Train classical ML models
│   ├── train_lstm_pytorch.py         # Train LSTM forecasters
│   ├── evaluate_classifier.py        # Model evaluation
│   └── check_predictions.py          # Inference testing
│
├── 📂 src/backend/                   # FastAPI application
│   ├── main.py                       # API entry point
│   ├── model_utils.py                # ML model utilities
│   ├── lstm_utils_pytorch.py         # LSTM inference
│   ├── routes/
│   │   └── generative.py             # AI chat & reports
│   └── static/                       # Dashboard frontend
│       ├── index.html                # Main dashboard
│       └── js/
│           ├── app.js                # Application logic
│           └── charts.js             # Chart.js visualizations
│
├── 📂 tests/                         # Test suite
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_api_health.py            # API tests
│   ├── test_model_utils.py           # Model tests
│   └── test_lstm_*.py                # LSTM tests
│
├── .env.example                      # Environment template
├── pyproject.toml                    # Project configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov=scripts --cov-report=html

# Type checking
mypy scripts/ src/ --ignore-missing-imports

# Linting
flake8 scripts/ src/

# Code formatting check
black scripts/ src/ --check
```

---

## ⚙️ Configuration

### Environment Variables

| Variable         | Default | Description                     |
| ---------------- | ------- | ------------------------------- |
| `GOOGLE_API_KEY` | -       | Google AI API key for AquaBot   |
| `SEED`           | 42      | Random seed for reproducibility |
| `DATA_DIR`       | data    | Data directory path             |
| `MODELS_DIR`     | models  | Models directory path           |
| `LOG_LEVEL`      | INFO    | Logging verbosity               |
| `CORS_ORIGINS`   | \*      | Allowed CORS origins            |

---

## 🛠️ Technology Stack

### Backend

- **FastAPI** - High-performance async web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Machine Learning

- **PyTorch** - Deep learning framework (LSTM)
- **scikit-learn** - Classical ML algorithms
- **pandas** - Data manipulation
- **NumPy** - Numerical computing

### AI/NLP

- **LangGraph** - AI agent framework
- **LangChain** - LLM integration
- **Google Generative AI** - Gemini models

### Frontend

- **Tailwind CSS** - Utility-first CSS framework
- **Chart.js** - Interactive charts
- **Web Speech API** - Voice input support

### DevOps

- **pytest** - Testing framework
- **mypy** - Static type checking
- **GitHub Actions** - CI/CD pipeline

---

## 🎓 Academic Context

This project was developed as a comprehensive demonstration of:

1. **Machine Learning Engineering**

   - End-to-end ML pipeline from data to deployment
   - Multiple model types (classification, regression, forecasting, anomaly detection)
   - Model evaluation and metrics

2. **Deep Learning**

   - LSTM sequence modeling for time-series forecasting
   - PyTorch implementation with training loops

3. **Software Engineering**

   - Clean architecture with separation of concerns
   - RESTful API design
   - Comprehensive testing

4. **Full-Stack Development**

   - Interactive web dashboard
   - Real-time updates and visualizations

5. **AI/LLM Integration**
   - Conversational AI with memory
   - Context-aware recommendations

---

## 📈 Future Enhancements

- [ ] Real sensor hardware integration (Arduino/Raspberry Pi)
- [ ] Mobile application (React Native)
- [ ] Multi-tenant support for multiple farms
- [ ] Advanced alerting (SMS, email notifications)
- [ ] Historical trend analysis and reporting
- [ ] Integration with weather APIs
- [ ] Automated model retraining pipeline

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Water Intelligence Project**  
Developed for aquaculture water quality monitoring and management.

---

<div align="center">

**🌊 Water Intelligence Project**

_Transforming aquaculture through AI-powered water quality management_

</div>

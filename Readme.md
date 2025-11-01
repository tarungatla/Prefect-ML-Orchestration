# 🚀 Bank Churn Prediction - ML Orchestration with Prefect

A production-ready machine learning pipeline for predicting bank customer churn, featuring automated data collection, model training, and continuous retraining orchestrated with **Prefect 3.0**.


## 🎯 Overview

This project demonstrates a complete ML orchestration system that:
- Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting)
- Automatically collects new data from user predictions
- Triggers model retraining when 500 new records are collected
- Provides comprehensive monitoring and health checks
- Deploys seamlessly with Prefect for production use

### System Architecture
```
┌─────────────────┐
│ Streamlit App   │ → Users input data
└────────┬────────┘
         ↓
    Saves to train.csv
         ↓
┌─────────────────────────────────────┐
│ Automated Collection Pipeline       │
│ (Runs every 60 seconds)            │
│ - Monitors train.csv               │
│ - Tracks count in data_counter.json│
└────────┬────────────────────────────┘
         ↓
   Count reaches 500?
         ↓ YES
┌─────────────────────────────────────┐
│ ML Training Pipeline                │
│ - Validates data                   │
│ - Engineers features               │
│ - Trains 3 models in parallel     │
│ - Selects best model               │
│ - Saves to models/                 │
└────────┬────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Streamlit loads latest model       │
│ Makes predictions                  │
└─────────────────────────────────────┘
```

## ✨ Features

### Core Capabilities
✅ **Data Pipeline**
- Automated data loading and validation
- Comprehensive quality checks
- Missing value handling
- Feature engineering

✅ **Model Training**
- Multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting)
- Parallel execution for faster training
- Cross-validation (5-fold)
- Automated model comparison and selection

✅ **Automated Retraining**
- Continuous data collection simulation
- Configurable retraining threshold (default: 500 records)
- Automatic model versioning
- Counter tracking and reset

✅ **Monitoring & Observability**
- Real-time health checks
- Model comparison reports
- Prefect UI dashboards

✅ **Production Features**
- Error handling and retry logic
- Model versioning with timestamps
- Scheduled execution support

## 📁 Project Structure
```
ml-orchestration-project/
│
├── ml_pipeline.py                    # Main ML training pipeline
├── automated_retraining_pipeline.py  # Auto collection & retraining

├── prefect.yaml                      # Deployment configuration
├── requirements.txt                  # Python dependencies
│
├── README.md                         # This file
├── data/                             # Data directory
│   ├── train.csv                     # Training data
│
├── models/                           # Saved models
│   ├── scaler.pkl                    # Feature scaler
│   └── *_model_*.pkl                 # Trained models
│
├── artifacts/                        # Pipeline artifacts
│   └── model_metadata_*.json         # Model metadata
│
└── data_counter.json                 # Retraining counter
```

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git (optional)

### Step 1: Clone or Download
```bash
# If using git
git clone https://github.com/tarungatla/Prefect-ML-Orchestration.git
cd ml-orchestration-project

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
# Check Prefect version
prefect version

# Check Python version
python --version
```

## 🚀 Quick Start


```bash
set PREFECT_API_URL=http://127.0.0.1:4200/api

prefect deploy -n ml_workflow_bank_churn

# Terminal 1: Start Prefect server
prefect server start

# Terminal 2: Start worker
prefect worker start --pool dev-pool

# Terminal 3: Run pipeline
python automated_retraining_pipeline.py
```



### View in Prefect UI

Access the dashboard at: **http://localhost:4200**

- View flow runs
- Monitor execution
- See beautiful markdown artifacts
- Track performance metr



## 📞 Support
- **Documentation**: Visit [docs.prefect.io](https://docs.prefect.io)

---



🎊 **Happy ML Orchestrating!**

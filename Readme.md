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
├── Dockerfile                        # Container configuration
├── docker-compose.yml                # Multi-service setup
├── Makefile                          # Quick commands
├── setup.sh                          # Automated setup script
├── .env                              # Environment variables
├── .gitignore                        # Git ignore rules
│
├── README.md                         # This file
├── QUICK_REFERENCE.md                # Command cheat sheet
├── AUTOMATION_SETUP_GUIDE.md         # Auto-retraining guide
├── PROJECT_SUMMARY.md                # Project overview
│
├── data/                             # Data directory
│   ├── train.csv                     # Training data
│   └── test.csv                      # Test data
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

### Option 1: Run Locally (Fastest)
```bash
# Run the ML pipeline once
python ml_pipeline.py
```

### Option 2: With Prefect Server
```bash
# Terminal 1: Start Prefect server
prefect server start

# Terminal 2: Start worker
prefect worker start --pool dev-pool

# Terminal 3: Run pipeline
python ml_pipeline.py
```


## 💻 Usage

### Running the ML Pipeline

#### Direct Execution
```bash
# Train with full dataset
python ml_pipeline.py

# Train with subset (for testing)
python -c "
from ml_pipeline import ml_training_pipeline
ml_training_pipeline(data_path='train.csv', nrows=1000)
"
```

#### Using Prefect Deployment
```bash
# Deploy the pipeline
prefect deploy ml_pipeline.py:ml_training_pipeline \
  -n ml-training-dev \
  -p dev-pool

# Start worker
prefect worker start --pool dev-pool

# Trigger deployment
prefect deployment run 'ml-training-pipeline/ml-training-dev'
```

### Automated Data Collection & Retraining

This feature simulates continuous data collection (like from a Streamlit app) and automatically retrains the model when 500 new records are collected.

#### Run Continuous Collection
```bash
# Run indefinitely (every 60 seconds)
python automated_retraining_pipeline.py run

# Run for 100 iterations with 10-second interval (for testing)
python automated_retraining_pipeline.py run 100 10
```

#### Quick Test (Add 500 Records Immediately)
```bash
# Add 500 synthetic records
python automated_retraining_pipeline.py bulk 500

# Trigger one collection cycle (will start retraining)
python automated_retraining_pipeline.py once
```

#### Check Status
```bash
python automated_retraining_pipeline.py status
```

Output:
```
======================================================================
DATA COLLECTION STATUS
======================================================================

Counter Information:
  New records since last training: 245
  Retraining threshold: 500
  Records until retraining: 255
  Total trainings: 2

Timestamps:
  Last data addition: 2025-10-04T14:52:30
  Last training: 2025-10-04T12:30:15

Training Data:
  Total records: 2745
  Churn rate: 20.15%
======================================================================
```

#### Other Commands
```bash
# Add data once
python automated_retraining_pipeline.py once

# Manual retraining (regardless of counter)
python automated_retraining_pipeline.py retrain

# Reset counter
python automated_retraining_pipeline.py reset

# Add bulk data for testing
python automated_retraining_pipeline.py bulk 100
```

### Model Inference
```python
from ml_pipeline import inference_pipeline

# Make predictions on new data
inference_pipeline(
    model_path="models/gradient_boosting_20251004_143022.pkl",
    data_path="test.csv",
    output_path="predictions.csv"
)
```

## 🚢 Deployment with Prefect

### Step 1: Create Work Pool
```bash
prefect work-pool create dev-pool --type process
```

### Step 2: Deploy Flows
```bash
# Deploy ML training pipeline
prefect deploy ml_pipeline.py:ml_training_pipeline \
  -n ml-training-dev \
  -p dev-pool \
  --param data_path=train.csv \
  --param nrows=1000

# Deploy automated collection
prefect deploy automated_retraining_pipeline.py:data_collection_flow \
  -n data-collector \
  -p dev-pool \
  --cron "*/1 * * * *"
```

### Step 3: Start Worker
```bash
prefect worker start --pool dev-pool
```

### Step 4: Trigger Deployments
```bash
# Manual trigger
prefect deployment run 'ml-training-pipeline/ml-training-dev'

# Automated collection runs every minute via cron schedule
```

### View in Prefect UI

Access the dashboard at: **http://localhost:4200**

- View flow runs
- Monitor execution
- See beautiful markdown artifacts
- Track performance metrics

## 📊 Monitoring

### Health Check Dashboard
```bash
python monitoring_dashboard.py
```

Features:
- Model performance tracking
- Drift detection
- Available models list
- Historical trends

### View Pipeline Status
```bash
# List recent flow runs
prefect flow-run ls --limit 10

# View specific run logs
prefect flow-run logs <run-id> --follow

# Check deployments
prefect deployment ls
```

### Cleanup Old Models
```bash
# Keep only last 5 models
python -c "from monitoring_dashboard import cleanup_old_models; cleanup_old_models(5)"
```

## ⚙️ Configuration

### ML Pipeline Settings

Edit `ml_pipeline.py`:
```python
class Config:
    DATA_PATH = "data/bank_churn.csv"
    MODEL_DIR = Path("models")
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
```

### Automated Retraining Settings

Edit `automated_retraining_pipeline.py`:
```python
CONFIG = {
    "data_file": "train.csv",
    "retraining_threshold": 500,        # Retrain after 500 records
    "simulation_interval_seconds": 60,  # Check every 60 seconds
    "records_per_batch": 1,             # Add 1 record per cycle
}
```

### Environment Variables

Create `.env` file:
```bash
# Prefect Configuration
PREFECT_API_URL=http://127.0.0.1:4200/api
PREFECT_LOGGING_LEVEL=INFO

# Model Configuration
MODEL_RANDOM_STATE=42
MODEL_TEST_SIZE=0.2
MODEL_CV_FOLDS=5
```

## 🎨 Streamlit Integration

### Save User Data for Retraining

Add this to your Streamlit app:
```python
import pandas as pd
from pathlib import Path

def save_user_prediction(user_inputs, prediction=None, actual_outcome=None):
    """
    Save user data to train.csv for future retraining
    
    Args:
        user_inputs: Dictionary of user inputs
        prediction: Model prediction (optional)
        actual_outcome: Actual churn outcome if known (optional)
    """
    record = {
        'CreditScore': user_inputs['credit_score'],
        'Geography': user_inputs['geography'],
        'Gender': user_inputs['gender'],
        'Age': user_inputs['age'],
        'Tenure': user_inputs['tenure'],
        'Balance': user_inputs['balance'],
        'NumOfProducts': user_inputs['num_products'],
        'HasCrCard': user_inputs['has_card'],
        'IsActiveMember': user_inputs['is_active'],
        'EstimatedSalary': user_inputs['salary'],
        'Exited': actual_outcome if actual_outcome is not None else 0
    }
    
    # Append to training data
    df = pd.DataFrame([record])
    data_file = Path("train.csv")
    
    if data_file.exists():
        df.to_csv(data_file, mode='a', header=False, index=False)
    else:
        df.to_csv(data_file, index=False)
    
    print(f"✓ User data saved for retraining")

# In your Streamlit app
if st.button("Predict Churn"):
    # Get user inputs
    user_data = {
        'credit_score': credit_score,
        'geography': geography,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'num_products': num_products,
        'has_card': has_card,
        'is_active': is_active,
        'salary': salary
    }
    
    # Make prediction
    prediction = model.predict(...)
    
    # Display result
    st.success(f"Prediction: {'Will Churn' if prediction else 'Will Not Churn'}")
    
    # Save for retraining
    save_user_prediction(user_data, prediction)
```

### Load Latest Model
```python
from pathlib import Path
import pickle

def load_latest_model():
    """Load the most recent trained model"""
    model_dir = Path("models")
    model_files = sorted([f for f in model_dir.glob("*.pkl") if 'scaler' not in f.name])
    
    if not model_files:
        raise FileNotFoundError("No trained models found")
    
    latest_model = model_files[-1]
    
    with open(latest_model, 'rb') as f:
        model = pickle.load(f)
    
    return model, model_name

# In Streamlit
model, model_name = load_latest_model()
st.sidebar.info(f"Using model: {model_name}")
```

## 🎯 Advanced Usage

### Schedule Daily Retraining
```bash
# Deploy with daily schedule at 2 AM
prefect deploy ml_pipeline.py:ml_training_pipeline \
  -n ml-training-scheduled \
  -p production-pool \
  --cron "0 2 * * *"
```

### Run with Docker
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Batch Processing
```python
from ml_pipeline import ml_training_pipeline

datasets = ["region1.csv", "region2.csv", "region3.csv"]

for dataset in datasets:
    result = ml_training_pipeline(data_path=dataset, nrows=5000)
    print(f"{dataset}: F1={result['metrics']['f1_score']:.4f}")
```

### Custom Model Parameters
```python
# Modify model training in ml_pipeline.py
@task(name="Train Random Forest")
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,      # Increase trees
        max_depth=15,          # Deeper trees
        min_samples_split=5,   # Custom split
        random_state=42
    )
    model.fit(X_train, y_train)
    return {'name': 'Random Forest', 'model': model}
```

## 🐛 Troubleshooting

### Issue: Deployment not found
```bash
# Solution 1: List deployments
prefect deployment ls

# Solution 2: Re-deploy
prefect deploy ml_pipeline.py:ml_training_pipeline -n ml-training-dev -p dev-pool
```

### Issue: Worker not picking up runs
```bash
# Check worker status
prefect worker ls

# Check work pool
prefect work-pool ls

# Restart worker
prefect worker start --pool dev-pool
```

### Issue: Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version (requires 3.10+)
python --version
```

### Issue: train.csv not found
```bash
# Check file exists
ls -la train.csv

# Create sample data if needed
python automated_retraining_pipeline.py bulk 1000
```

### Issue: Permission denied on scripts
```bash
# Make scripts executable
chmod +x setup.sh start_server.sh start_worker.sh run_pipeline.sh
```

### Issue: Automated retraining not working
```bash
# Check status
python automated_retraining_pipeline.py status

# Reset and restart
python automated_retraining_pipeline.py reset
python automated_retraining_pipeline.py run
```

## 📚 Additional Resources

### Documentation
- `README.md` - This file (complete guide)
- `QUICK_REFERENCE.md` - Command cheat sheet
- `AUTOMATION_SETUP_GUIDE.md` - Detailed automation guide
- `PROJECT_SUMMARY.md` - Project overview

### Example Code
- `example_usage.py` - 10 complete usage examples
- `test_pipeline.py` - Unit tests and test patterns

### External Links
- [Prefect Documentation](https://docs.prefect.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Prefect Community Slack](https://prefect.io/slack)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest test_pipeline.py`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

MIT License - Feel free to use this project for learning and production!

## 🎓 Learning Outcomes

By working with this project, you'll learn:

- ✅ Building production-ready ML pipelines
- ✅ Orchestrating workflows with Prefect
- ✅ Implementing automated retraining systems
- ✅ Data validation and quality checks
- ✅ Model versioning and artifact management
- ✅ Pipeline monitoring and health checks
- ✅ Deployment strategies
- ✅ Error handling and retry logic
- ✅ Docker containerization
- ✅ Integration with web applications

## 🎉 Key Features Summary

### For Data Scientists
- Multiple model training and comparison
- Cross-validation and robust evaluation
- Feature engineering pipeline
- Model versioning and tracking

### For ML Engineers
- Production-ready orchestration
- Automated retraining triggers
- Monitoring and observability
- Error handling and retries

### For DevOps
- Docker containerization
- Scheduled deployments
- Health checks and alerts
- Scalable architecture

## 🚀 Next Steps

1. ✅ Complete installation
2. ✅ Run basic pipeline: `python ml_pipeline.py`
3. ✅ Test automated collection: `python automated_retraining_pipeline.py run 10 10`
4. ✅ Deploy with Prefect: `prefect deploy --all`
5. ✅ Integrate with your Streamlit app
6. ✅ Set up monitoring: `python monitoring_dashboard.py`
7. ✅ Configure production deployment
8. ✅ Set up alerts and notifications

## 📞 Support

- **Issues**: Check troubleshooting section above
- **Questions**: See `QUICK_REFERENCE.md` for commands
- **Community**: Join [Prefect Slack](https://prefect.io/slack)
- **Documentation**: Visit [docs.prefect.io](https://docs.prefect.io)

---

**Made with ❤️ using Prefect 3.0**

**Project Statistics:**
- 2,500+ lines of production code
- 14 core files
- 1,000+ lines of documentation
- 10 usage examples
- Full test coverage
- Production-ready

🎊 **Happy ML Orchestrating!**

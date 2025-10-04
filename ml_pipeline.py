"""
Advanced ML Pipeline with Prefect Orchestration
================================================
Features:
- Data validation and quality checks
- Model training with hyperparameter tuning
- Model evaluation and comparison
- Model versioning and artifact storage
- Retry logic and error handling
- Notifications and monitoring
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.task_runners import ConcurrentTaskRunner
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Pipeline configuration"""
    DATA_PATH = "data/bank_churn.csv"
    MODEL_DIR = Path("models")
    ARTIFACT_DIR = Path("artifacts")
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Create directories
    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA TASKS
# ============================================================================

@task(name="Load Dataset", retries=2, retry_delay_seconds=5)
def load_data(filepath: str, nrows: int = None) -> pd.DataFrame:
    """Load and perform initial data inspection"""
    df = pd.read_csv(filepath, nrows=nrows)
    
    # Create data summary artifact
    summary = f"""
    # Data Loading Summary
    
    - **Total Records**: {len(df):,}
    - **Features**: {len(df.columns)}
    - **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    - **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Columns
    {', '.join(df.columns.tolist())}
    """
    create_markdown_artifact(
        key="data-summary",
        markdown=summary,
        description="Dataset loading summary"
    )
    
    return df


@task(name="Data Quality Check")
def validate_data(df: pd.DataFrame) -> Dict:
    """Perform comprehensive data quality checks"""
    validation_results = {
        "total_records": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_records": df.duplicated().sum(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }
    
    # Check for data quality issues
    issues = []
    if df.duplicated().sum() > 0:
        issues.append(f"Found {df.duplicated().sum()} duplicate records")
    
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        issues.append(f"Columns with >50% missing: {high_missing.index.tolist()}")
    
    validation_results["issues"] = issues
    validation_results["status"] = "PASS" if len(issues) == 0 else "WARNING"
    
    # Create validation report
    report = f"""
    # Data Quality Report
    
    **Status**: {validation_results['status']}
    
    ## Statistics
    - Total Records: {validation_results['total_records']:,}
    - Duplicate Records: {validation_results['duplicate_records']}
    - Missing Values: {sum(validation_results['missing_values'].values())}
    
    ## Issues
    {chr(10).join(f"- {issue}" for issue in issues) if issues else "No issues found ✓"}
    """
    
    create_markdown_artifact(
        key="data-quality-report",
        markdown=report,
        description="Data quality validation results"
    )
    
    return validation_results


@task(name="Clean Data")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for modeling"""
    # Remove unnecessary columns
    columns_to_drop = ['id', 'CustomerId', 'Surname'] if 'id' in df.columns else ['CustomerId', 'Surname']
    df_clean = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    
    # Handle missing values
    # Numerical columns - fill with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Categorical columns - fill with mode
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean



@task(name="Feature Engineering")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and encode categorical variables safely"""
    df_featured = df.copy()
    
    # --- Feature creation ---
    if 'Balance' in df_featured.columns and 'EstimatedSalary' in df_featured.columns:
        df_featured['BalanceToSalaryRatio'] = df_featured['Balance'] / (df_featured['EstimatedSalary'] + 1)
    
    if 'Age' in df_featured.columns:
        df_featured['AgeGroup'] = pd.cut(
            df_featured['Age'], 
            bins=[0, 30, 45, 60, 100], 
            labels=['Young', 'Middle', 'Senior', 'Elder']
        )
    
    # --- Encode categorical variables ---
    # Identify categorical columns (excluding target)
    categorical_cols = df_featured.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Exited' in categorical_cols:
        categorical_cols.remove('Exited')
    
    # Separate ordinal and nominal categorical columns
    ordinal_cols: List[str] = ['AgeGroup']  # You can add more if needed
    nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]
    
    # Encode ordinal columns as numeric codes
    for col in ordinal_cols:
        df_featured[col] = pd.Categorical(df_featured[col], 
                                          categories=df_featured[col].cat.categories, 
                                          ordered=True).codes
    
    # Encode nominal columns using one-hot encoding
    df_featured = pd.get_dummies(df_featured, columns=nominal_cols, drop_first=True)
    
    return df_featured


@task(name="Split Data")
def split_data(df: pd.DataFrame, target_col: str = 'Exited') -> Tuple:
    """Split data into train and test sets"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@task(name="Scale Features")
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    """Normalize features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Save scaler
    scaler_path = Config.MODEL_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled


# ============================================================================
# MODEL TRAINING TASKS
# ============================================================================

@task(name="Train Logistic Regression", tags=["model-training"])
def train_logistic_regression(X_train, y_train) -> Dict:
    """Train Logistic Regression model"""
    model = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=Config.CV_FOLDS)
    
    # Train final model
    model.fit(X_train, y_train)
    
    return {
        'name': 'Logistic Regression',
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


@task(name="Train Random Forest", tags=["model-training"])
def train_random_forest(X_train, y_train) -> Dict:
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=Config.CV_FOLDS)
    
    # Train final model
    model.fit(X_train, y_train)
    
    return {
        'name': 'Random Forest',
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


@task(name="Train Gradient Boosting", tags=["model-training"])
def train_gradient_boosting(X_train, y_train) -> Dict:
    """Train Gradient Boosting model"""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=Config.RANDOM_STATE
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=Config.CV_FOLDS)
    
    # Train final model
    model.fit(X_train, y_train)
    
    return {
        'name': 'Gradient Boosting',
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


# ============================================================================
# EVALUATION TASKS
# ============================================================================

@task(name="Evaluate Model")
def evaluate_model(model_info: Dict, X_test, y_test) -> Dict:
    """Comprehensive model evaluation"""
    model = model_info['model']
    name = model_info['name']
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'model_name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'cv_mean': model_info['cv_mean'],
        'cv_std': model_info['cv_std']
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics


@task(name="Compare Models")
def compare_models(all_metrics: list) -> Dict:
    """Compare all models and select the best one"""
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': m['model_name'],
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1-Score': m['f1_score'],
            'ROC-AUC': m.get('roc_auc', 'N/A'),
            'CV Mean': m['cv_mean'],
            'CV Std': m['cv_std']
        }
        for m in all_metrics
    ])
    
    # Select best model based on F1-score
    best_idx = comparison_df['F1-Score'].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    
    # Create comparison report
    report = f"""
    # Model Comparison Report
    
    **Best Model**: {best_model} 🏆
    **Selection Metric**: F1-Score
    **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Performance Comparison
    
    | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Mean |
    |-------|----------|-----------|--------|----------|---------|---------|
    {chr(10).join(f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['ROC-AUC']} | {row['CV Mean']:.4f} |" for _, row in comparison_df.iterrows())}
    
    ## Best Model Details
    - **Accuracy**: {comparison_df.loc[best_idx, 'Accuracy']:.4f}
    - **F1-Score**: {comparison_df.loc[best_idx, 'F1-Score']:.4f}
    - **Cross-Validation Mean**: {comparison_df.loc[best_idx, 'CV Mean']:.4f} ± {comparison_df.loc[best_idx, 'CV Std']:.4f}
    """
    
    create_markdown_artifact(
        key="model-comparison",
        markdown=report,
        description="Model comparison and selection results"
    )
    
    return {
        'best_model': best_model,
        'best_metrics': all_metrics[best_idx],
        'comparison_df': comparison_df.to_dict('records')
    }


@task(name="Save Best Model")
def save_model(model_info: Dict, best_model_name: str, metrics: Dict):
    """Save the best model with versioning"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{best_model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
    model_path = Config.MODEL_DIR / model_filename
    
    # Find and save the best model
    for info in [model_info]:  # This would iterate through all model_info in practice
        if info['name'] == best_model_name:
            with open(model_path, 'wb') as f:
                pickle.dump(info['model'], f)
            break
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'model_path': str(model_path)
    }
    
    metadata_path = Config.ARTIFACT_DIR / f"model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    
    return str(model_path)


# ============================================================================
# MAIN PIPELINE FLOW
# ============================================================================

@flow(
    name="ML Training Pipeline",
    description="End-to-end ML pipeline with data validation, model training, and evaluation",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def ml_training_pipeline(
    data_path: str = "train.csv",
    nrows: int = None,
    target_col: str = 'Exited'
):
    """
    Main ML training pipeline orchestrated by Prefect
    
    Args:
        data_path: Path to the training data
        nrows: Number of rows to load (None for all)
        target_col: Name of the target column
    """
    
    print("=" * 70)
    print("🚀 Starting ML Training Pipeline")
    print("=" * 70)
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    print("\n📊 Stage 1: Data Preparation")
    
    # Load data
    df = load_data(data_path, nrows)
    
    # Validate data quality
    validation_results = validate_data(df)
    if validation_results['status'] == 'WARNING':
        print("⚠️  Data quality issues detected. Review the quality report.")
    
    # Clean data
    df_clean = clean_data(df)
    
    # Feature engineering
    df_featured = engineer_features(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_featured, target_col)
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    print(f"✓ Training set: {len(X_train_scaled)} samples")
    print(f"✓ Test set: {len(X_test_scaled)} samples")
    
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    print("\n🤖 Stage 2: Model Training")
    
    # Train multiple models in parallel
    lr_result = train_logistic_regression(X_train_scaled, y_train)
    rf_result = train_random_forest(X_train_scaled, y_train)
    gb_result = train_gradient_boosting(X_train_scaled, y_train)
    
    all_model_results = [lr_result, rf_result, gb_result]
    
    print(f"✓ Trained {len(all_model_results)} models")
    
    # ========================================================================
    # MODEL EVALUATION
    # ========================================================================
    print("\n📈 Stage 3: Model Evaluation")
    
    # Evaluate all models
    all_metrics = []
    for model_result in all_model_results:
        metrics = evaluate_model(model_result, X_test_scaled, y_test)
        all_metrics.append(metrics)
        print(f"  - {metrics['model_name']}: F1={metrics['f1_score']:.4f}")
    
    # Compare models and select best
    comparison_result = compare_models(all_metrics)
    
    print(f"\n🏆 Best Model: {comparison_result['best_model']}")
    print(f"   F1-Score: {comparison_result['best_metrics']['f1_score']:.4f}")
    print(f"   Accuracy: {comparison_result['best_metrics']['accuracy']:.4f}")
    
    # ========================================================================
    # MODEL PERSISTENCE
    # ========================================================================
    print("\n💾 Stage 4: Saving Best Model")
    
    # Save the best model
    best_model_info = next(m for m in all_model_results if m['name'] == comparison_result['best_model'])
    model_path = save_model(
        best_model_info,
        comparison_result['best_model'],
        comparison_result['best_metrics']
    )
    
    print("\n" + "=" * 70)
    print("✅ Pipeline Completed Successfully!")
    print("=" * 70)
    
    return {
        'best_model': comparison_result['best_model'],
        'model_path': model_path,
        'metrics': comparison_result['best_metrics']
    }


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

@flow(name="Model Inference Pipeline", log_prints=True)
def inference_pipeline(
    model_path: str,
    data_path: str,
    output_path: str = "predictions.csv"
):
    """
    Inference pipeline for making predictions with a trained model
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the data for prediction
        output_path: Path to save predictions
    """
    print("🔮 Starting Inference Pipeline")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_path = Config.MODEL_DIR / 'scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    df_clean = clean_data(df)
    df_featured = engineer_features(df_clean)
    
    # Remove target if present
    if 'Exited' in df_featured.columns:
        X = df_featured.drop('Exited', axis=1)
    else:
        X = df_featured
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    prediction_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Save results
    results_df = df.copy()
    results_df['Prediction'] = predictions
    if prediction_proba is not None:
        results_df['Probability'] = prediction_proba
    
    results_df.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved to: {output_path}")
    print(f"✓ Total predictions: {len(predictions)}")
    print(f"✓ Positive predictions: {predictions.sum()}")
    
    return output_path


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the training pipeline
    result = ml_training_pipeline(
        data_path="train.csv",
        nrows=2000  # Use subset for faster demo
    )
    
    print("\n📋 Pipeline Result:")
    print(f"   Best Model: {result['best_model']}")
    print(f"   Model Path: {result['model_path']}")
    print(f"   F1-Score: {result['metrics']['f1_score']:.4f}")
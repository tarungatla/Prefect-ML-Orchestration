"""
Automated Data Collection and Model Retraining Pipeline
========================================================
Simulates continuous data collection and triggers retraining
when threshold is reached.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash

from ml_pipeline import ml_training_pipeline

# Configuration
CONFIG = {
    "data_file": "train.csv",
    "counter_file": "data_counter.json",
    "retraining_threshold": 500,
    "simulation_interval_seconds": 60,  # Every 1 minute
    "records_per_batch": 50,  # Add 1 record per minute
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_counter() -> Dict:
    """Load the data counter from file"""
    counter_file = Path(CONFIG["counter_file"])
    
    if counter_file.exists():
        with open(counter_file, 'r') as f:
            return json.load(f)
    
    return {
        "total_new_records": 0,
        "last_training_timestamp": None,
        "last_data_addition": None,
        "training_count": 0
    }


def save_counter(counter: Dict):
    """Save the data counter to file"""
    with open(CONFIG["counter_file"], 'w') as f:
        json.dump(counter, f, indent=4)


def generate_synthetic_record() -> Dict:
    """Generate a single synthetic customer record"""
    np.random.seed(int(time.time() * 1000) % 2**32)
    
    record = {
        'CreditScore': np.random.randint(300, 850),
        'Geography': np.random.choice(['France', 'Germany', 'Spain']),
        'Gender': np.random.choice(['Male', 'Female']),
        'Age': np.random.randint(18, 80),
        'Tenure': np.random.randint(0, 10),
        'Balance': round(np.random.uniform(0, 250000), 2),
        'NumOfProducts': np.random.randint(1, 5),
        'HasCrCard': np.random.randint(0, 2),
        'IsActiveMember': np.random.randint(0, 2),
        'EstimatedSalary': round(np.random.uniform(10000, 200000), 2),
        'Exited': np.random.choice([0, 1], p=[0.8, 0.2])  # 20% churn rate
    }
    
    return record


# ============================================================================
# TASKS
# ============================================================================

@task(name="Check Data Counter", cache_key_fn=task_input_hash, cache_expiration=timedelta(seconds=30))
def check_counter() -> Dict:
    """Check current counter status"""
    counter = load_counter()
    print(f"Current new records: {counter['total_new_records']}/{CONFIG['retraining_threshold']}")
    return counter


@task(name="Generate New Data")
def generate_new_data(num_records: int = 1) -> pd.DataFrame:
    """Generate synthetic new data records"""
    records = [generate_synthetic_record() for _ in range(num_records)]
    df = pd.DataFrame(records)
    
    print(f"Generated {len(df)} new records")
    return df


@task(name="Append to Training Data")
def append_to_training_data(new_data: pd.DataFrame) -> int:
    """Append new data to the training CSV file"""
    data_file = Path(CONFIG["data_file"])
    
    if not data_file.exists():
        print(f"Warning: {CONFIG['data_file']} not found. Creating new file.")
        new_data.to_csv(data_file, index=False)
    else:
        # Append without header
        new_data.to_csv(data_file, mode='a', header=False, index=False)
    
    print(f"Appended {len(new_data)} records to {CONFIG['data_file']}")
    return len(new_data)


@task(name="Update Counter")
def update_counter(records_added: int) -> Dict:
    """Update the counter with new records"""
    counter = load_counter()
    counter["total_new_records"] += records_added
    counter["last_data_addition"] = datetime.now().isoformat()
    save_counter(counter)
    
    print(f"Counter updated: {counter['total_new_records']} new records total")
    return counter


@task(name="Check Retraining Threshold")
def should_retrain(counter: Dict) -> bool:
    """Check if we should trigger retraining"""
    threshold_reached = counter["total_new_records"] >= CONFIG["retraining_threshold"]
    
    if threshold_reached:
        print(f"Threshold reached! {counter['total_new_records']} >= {CONFIG['retraining_threshold']}")
    else:
        remaining = CONFIG["retraining_threshold"] - counter["total_new_records"]
        print(f"Threshold not reached. Need {remaining} more records.")
    
    return threshold_reached


@task(name="Reset Counter")
def reset_counter():
    """Reset counter after retraining"""
    counter = load_counter()
    counter["total_new_records"] = 0
    counter["last_training_timestamp"] = datetime.now().isoformat()
    counter["training_count"] += 1
    save_counter(counter)
    
    print(f"Counter reset. Total trainings: {counter['training_count']}")


@task(name="Trigger Model Retraining")
def trigger_retraining():
    """Trigger the ML training pipeline"""
    print("=" * 70)
    print("TRIGGERING MODEL RETRAINING")
    print("=" * 70)
    
    try:
        result = ml_training_pipeline(
            data_path=CONFIG["data_file"],
            nrows=None  # Use all data for retraining
        )
        
        print(f"\nRetraining completed successfully!")
        print(f"Best Model: {result['best_model']}")
        print(f"F1-Score: {result['metrics']['f1_score']:.4f}")
        print(f"Model saved to: {result['model_path']}")
        
        return result
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        raise


# ============================================================================
# FLOWS
# ============================================================================

@flow(name="Data Collection Flow", log_prints=True)
def data_collection_flow():
    """
    Simulates continuous data collection by adding records to train.csv
    """
    print("\n" + "=" * 70)
    print(f"DATA COLLECTION CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check current counter
    counter = check_counter()
    
    # Generate new data
    new_data = generate_new_data(num_records=CONFIG["records_per_batch"])
    
    # Append to training file
    records_added = append_to_training_data(new_data)
    
    # Update counter
    updated_counter = update_counter(records_added)
    
    # Check if we should retrain
    needs_retraining = should_retrain(updated_counter)
    
    if needs_retraining:
        print("\nINITIATING AUTOMATIC RETRAINING...")
        
        # Trigger retraining
        retraining_result = trigger_retraining()
        
        # Reset counter
        reset_counter()
        
        return {
            "status": "retrained",
            "records_added": records_added,
            "model": retraining_result['best_model'],
            "f1_score": retraining_result['metrics']['f1_score']
        }
    
    return {
        "status": "data_added",
        "records_added": records_added,
        "total_new_records": updated_counter["total_new_records"],
        "records_until_retraining": CONFIG["retraining_threshold"] - updated_counter["total_new_records"]
    }


@flow(name="Continuous Data Pipeline", log_prints=True)
def continuous_data_pipeline(num_iterations: int = None, interval_seconds: int = None):
    """
    Runs the data collection flow continuously
    
    Args:
        num_iterations: Number of iterations to run (None = infinite)
        interval_seconds: Seconds between iterations (default from CONFIG)
    """
    interval = interval_seconds or CONFIG["simulation_interval_seconds"]
    iteration = 0
    
    print("=" * 70)
    print("STARTING CONTINUOUS DATA COLLECTION PIPELINE")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Interval: {interval} seconds")
    print(f"  - Records per batch: {CONFIG['records_per_batch']}")
    print(f"  - Retraining threshold: {CONFIG['retraining_threshold']} records")
    print(f"  - Target: {CONFIG['data_file']}")
    print("=" * 70)
    
    try:
        while True:
            iteration += 1
            
            if num_iterations and iteration > num_iterations:
                print(f"\nCompleted {num_iterations} iterations. Stopping.")
                break
            
            print(f"\n--- Iteration {iteration} ---")
            
            # Run data collection
            result = data_collection_flow()
            
            if result["status"] == "retrained":
                print(f"\nModel retrained! New model: {result['model']}")
            else:
                print(f"\nData added. {result['records_until_retraining']} records until retraining.")
            
            # Wait for next iteration
            if num_iterations is None or iteration < num_iterations:
                print(f"\nWaiting {interval} seconds until next cycle...")
                time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nPipeline stopped by user.")
    except Exception as e:
        print(f"\n\nPipeline error: {e}")
        raise


@flow(name="Manual Retraining Trigger", log_prints=True)
def manual_retraining_flow():
    """
    Manually trigger retraining regardless of counter
    """
    print("=" * 70)
    print("MANUAL RETRAINING TRIGGERED")
    print("=" * 70)
    
    result = trigger_retraining()
    reset_counter()
    
    return result


@flow(name="View Status", log_prints=True)
def view_status_flow():
    """
    View current status of data collection and counter
    """
    counter = load_counter()
    data_file = Path(CONFIG["data_file"])
    
    print("=" * 70)
    print("DATA COLLECTION STATUS")
    print("=" * 70)
    
    # Counter info
    print(f"\nCounter Information:")
    print(f"  New records since last training: {counter['total_new_records']}")
    print(f"  Retraining threshold: {CONFIG['retraining_threshold']}")
    print(f"  Records until retraining: {CONFIG['retraining_threshold'] - counter['total_new_records']}")
    print(f"  Total trainings: {counter['training_count']}")
    
    # Timestamps
    print(f"\nTimestamps:")
    if counter['last_data_addition']:
        print(f"  Last data addition: {counter['last_data_addition']}")
    if counter['last_training_timestamp']:
        print(f"  Last training: {counter['last_training_timestamp']}")
    
    # Data file info
    if data_file.exists():
        df = pd.read_csv(data_file)
        print(f"\nTraining Data:")
        print(f"  Total records: {len(df)}")
        print(f"  Churn rate: {df['Exited'].mean():.2%}" if 'Exited' in df.columns else "")
    else:
        print(f"\nWarning: {CONFIG['data_file']} not found!")
    
    print("=" * 70)
    
    return counter


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def reset_system():
    """Reset the entire system (counter and optionally data)"""
    print("Resetting system...")
    
    counter = {
        "total_new_records": 0,
        "last_training_timestamp": None,
        "last_data_addition": None,
        "training_count": 0
    }
    save_counter(counter)
    
    print("Counter reset complete.")


def add_bulk_data(num_records: int):
    """Add bulk data for testing"""
    print(f"Adding {num_records} records...")
    
    new_data = generate_new_data(num_records)
    records_added = append_to_training_data(new_data)
    update_counter(records_added)
    
    print(f"Added {records_added} records successfully.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "run":
            # Run continuous pipeline
            iterations = int(sys.argv[2]) if len(sys.argv) > 2 else None
            interval = int(sys.argv[3]) if len(sys.argv) > 3 else None
            continuous_data_pipeline(num_iterations=iterations, interval_seconds=interval)
            
        elif command == "once":
            # Run one iteration
            data_collection_flow()
            
        elif command == "retrain":
            # Manual retraining
            manual_retraining_flow()
            
        elif command == "status":
            # View status
            view_status_flow()
            
        elif command == "reset":
            # Reset system
            reset_system()
            
        elif command == "bulk":
            # Add bulk data
            num = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            add_bulk_data(num)
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: run, once, retrain, status, reset, bulk")
    else:
        print("\nAutomated Data Collection & Retraining Pipeline")
        print("=" * 70)
        print("\nUsage:")
        print("  python automated_retraining_pipeline.py run [iterations] [interval]")
        print("  python automated_retraining_pipeline.py once")
        print("  python automated_retraining_pipeline.py retrain")
        print("  python automated_retraining_pipeline.py status")
        print("  python automated_retraining_pipeline.py reset")
        print("  python automated_retraining_pipeline.py bulk [num_records]")
        print("\nExamples:")
        print("  # Run continuously (Ctrl+C to stop)")
        print("  python automated_retraining_pipeline.py run")
        print("\n  # Run for 10 iterations with 30 second interval")
        print("  python automated_retraining_pipeline.py run 10 30")
        print("\n  # Add data once")
        print("  python automated_retraining_pipeline.py once")
        print("\n  # View status")
        print("  python automated_retraining_pipeline.py status")
        print("\n  # Add 500 records at once (for testing)")
        print("  python automated_retraining_pipeline.py bulk 500")
"""
Airflow DAG for Handwritten Digit Recognition MLOps Pipeline
Automated pipeline running every 5 minutes
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

# Default arguments
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'digit_recognition_mlops_pipeline',
    default_args=default_args,
    description='Automated MLOps pipeline for handwritten digit recognition',
    schedule_interval=timedelta(minutes=5),  # Run every 5 minutes
    catchup=False,
    tags=['mlops', 'digits', 'recognition', 'prediction'],
)

def data_engineering_task():
    """Data engineering stage: load, clean, and split data"""
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.append(project_root)
    
    from code.datasets.data_pipeline import main as data_pipeline_main
    
    print("Starting data engineering stage...")
    train_path, test_path = data_pipeline_main()
    print(f"Data engineering completed. Train: {train_path}, Test: {test_path}")
    
    return {"train_path": train_path, "test_path": test_path}

def model_engineering_task():
    """Model engineering stage: train, evaluate, and package model"""
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.append(project_root)
    
    from code.models.model_pipeline import main as model_pipeline_main
    
    print("Starting model engineering stage...")
    model, metrics = model_pipeline_main()
    print(f"Model engineering completed. Accuracy: {metrics['accuracy']:.4f}")
    
    return {"model_accuracy": metrics['accuracy'], "model_path": "../../models/digits_model.pkl"}

def deployment_task():
    """Deployment stage: build and run Docker containers"""
    import subprocess
    import os
    
    print("Starting deployment stage...")
    
    # Change to deployment directory
    deployment_dir = os.path.join(os.path.dirname(__file__), '../../..', 'code', 'deployment')
    os.chdir(deployment_dir)
    
    try:
        # Stop existing containers
        print("Stopping existing containers...")
        subprocess.run(['docker-compose', 'down'], check=True, capture_output=True)
        
        # Build and start containers
        print("Building and starting containers...")
        result = subprocess.run(
            ['docker-compose', 'up', '--build', '-d'], 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        print("Deployment completed successfully!")
        print("API available at: http://localhost:8000")
        print("App available at: http://localhost:8501")
        
        return {"status": "success", "api_url": "http://localhost:8000", "app_url": "http://localhost:8501"}
        
    except subprocess.CalledProcessError as e:
        print(f"Deployment failed: {e}")
        print(f"Error output: {e.stderr}")
        raise

def health_check_task():
    """Health check: verify that API and app are running"""
    import requests
    import time
    
    print("Starting health check...")
    
    # Wait a bit for containers to start
    time.sleep(30)
    
    try:
        # Check API health
        api_response = requests.get("http://localhost:8000/health", timeout=10)
        api_healthy = api_response.status_code == 200
        
        # Check if app is accessible (basic check)
        app_response = requests.get("http://localhost:8501", timeout=10)
        app_healthy = app_response.status_code == 200
        
        if api_healthy and app_healthy:
            print("✅ Health check passed: Both API and app are running")
            return {"api_healthy": True, "app_healthy": True}
        else:
            print(f"❌ Health check failed: API={api_healthy}, App={app_healthy}")
            return {"api_healthy": api_healthy, "app_healthy": app_healthy}
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return {"api_healthy": False, "app_healthy": False, "error": str(e)}

# Task definitions
data_engineering = PythonOperator(
    task_id='data_engineering',
    python_callable=data_engineering_task,
    dag=dag,
)

model_engineering = PythonOperator(
    task_id='model_engineering',
    python_callable=model_engineering_task,
    dag=dag,
)

deployment = PythonOperator(
    task_id='deployment',
    python_callable=deployment_task,
    dag=dag,
)

health_check = PythonOperator(
    task_id='health_check',
    python_callable=health_check_task,
    dag=dag,
)

# Task dependencies
data_engineering >> model_engineering >> deployment >> health_check

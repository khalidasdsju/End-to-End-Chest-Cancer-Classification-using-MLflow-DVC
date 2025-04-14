import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
from mlflow.exceptions import MlflowException

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("https://dagshub.com/khalidasdsju/End-to-End-Chest-Cancer-Classification-using-MLflow-DVC.mlflow")
    os.environ['MLFLOW_TRACKING_USERNAME'] = "khalidasdsju"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "55e0fd18547b3cb185580e2c4f3932fd668f1257"

def get_latest_version():
    """Get the latest model version"""
    client = mlflow.tracking.MlflowClient()
    try:
        all_versions = client.search_model_versions("name='random_forest_model'")
        if not all_versions:
            return 0
        return max(int(mv.version) for mv in all_versions)
    except:
        return 0

def train_regression_model():
    """Train and log random forest regression model with MLflow"""
    # Generate synthetic regression data
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Set up MLflow experiment
    mlflow.set_experiment("random_forest_regression_experiment")

    # Get the next version number
    next_version = get_latest_version() + 1

    # Define model parameters
    params = {
        "n_estimators": 1000,
        "criterion": "squared_error",
        "max_depth": None,
        "min_samples_split": 15,
        "min_samples_leaf": 5,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": None,
        "random_state": None,
        "verbose": 0,
        "warm_start": False,
        "ccp_alpha": 0.0,
        "max_samples": None
    }

    # Start MLflow run
    with mlflow.start_run() as run:
        try:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_version", next_version)
            
            # Train the model
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Log metrics
            metrics = {
                "mse": mse,
                "rmse": rmse
            }
            mlflow.log_metrics(metrics)
            
            # Log the model with version
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="random_forest_model"
            )
            
            # Print run info
            print(f"\n=== MLflow Run ID: {run.info.run_id} ===")
            print(f"Model Version: {next_version}")
            print(f"Artifact URI: {mlflow.get_artifact_uri()}")
            
            # Print metrics
            print("\n=== Model Performance Metrics ===")
            for metric_name, value in metrics.items():
                print(f"{metric_name.upper()}: {value:.4f}")

            return model, metrics

        except MlflowException as e:
            print(f"MLflow error occurred: {str(e)}")
            raise
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

if __name__ == "__main__":
    setup_mlflow()
    model, metrics = train_regression_model()
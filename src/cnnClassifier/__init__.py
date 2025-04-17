import os
import sys
import logging


# Set MLFlow environment variables
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/khalidasdsju/End-to-End-Chest-Cancer-Classification-using-MLflow-DVC.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "khalidasdsju"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "deb2ac32c761ce9c633f67bf1076091df610d6c7"

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cnnClassifierLogger")
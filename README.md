# End-to-End Chest Cancer Classification using MLflow & DVC

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange) ![MLflow](https://img.shields.io/badge/MLflow-2.2.2-green) ![DVC](https://img.shields.io/badge/DVC-latest-purple) ![Flask](https://img.shields.io/badge/Flask-latest-lightgrey)

A deep learning project for chest cancer classification from CT scan images with MLflow tracking and DVC versioning.

## Project Overview

This project implements an end-to-end machine learning pipeline for classifying chest CT scan images as either normal or cancerous. It includes data ingestion, model training with VGG16, experiment tracking, and a web application for predictions.

## Features

- Modular pipeline architecture with clear separation of concerns
- Transfer learning with VGG16 for chest cancer classification
- Experiment tracking with MLflow and data versioning with DVC
- Interactive Flask web application for real-time predictions
- Configurable hyperparameters via params.yaml

## Installation

```bash
# Clone repository
git clone https://github.com/khalidasdsju/End-to-End-Chest-Cancer-Classification-using-MLflow-DVC.git
cd End-to-End-Chest-Cancer-Classification-using-MLflow-DVC

# Create and activate virtual environment
python -m venv chest
source chest/bin/activate  # On Windows: chest\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## MLflow Configuration

Create a `.env` file with the following content:

```
MLFLOW_TRACKING_URI=https://dagshub.com/khalidasdsju/End-to-End-Chest-Cancer-Classification-using-MLflow-DVC.mlflow
MLFLOW_TRACKING_USERNAME=khalidasdsju
MLFLOW_TRACKING_PASSWORD=deb2ac32c761ce9c633f67bf1076091df610d6c7
```

## Usage

### Training Pipeline

```bash
# Run complete pipeline
python main.py

# Run individual stages with DVC
dvc repro data_ingestion
dvc repro prepare_base_model
dvc repro training
dvc repro evaluation
```

### Web Application

```bash
# Start the application
cd app
python app.py
# Or use the script
./start_app.sh
```

Access the application at http://127.0.0.1:8080

## Project Structure

```
├── app/                      # Web application
├── artifacts/                # Generated artifacts
├── config/                   # Configuration files
├── logs/                     # Application logs
├── research/                 # Research notebooks
├── src/                      # Source code
│   └── cnnClassifier/        # Main package
├── .dvc/                     # DVC configuration
├── .env                      # Environment variables
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # Model parameters
├── requirements.txt          # Project dependencies
└── setup.py                  # Package setup
```

## Pipeline Components

1. **Data Ingestion**: Downloads and extracts the chest CT scan dataset
2. **Base Model Preparation**: Prepares VGG16-based architecture with custom layers
3. **Model Training**: Trains with data augmentation and custom parameters
4. **Model Evaluation**: Evaluates performance and logs metrics to MLflow

## Model Parameters

Parameters are configurable in `params.yaml`:

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 15
INCLUDE_TOP: False
EPOCHS: 2
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.015
```

## Web Application

The Flask application allows users to:
- Upload chest CT scan images
- Get predictions (normal or cancer) with confidence scores
- View visual explanations of the model's decision

## Results

The model achieves approximately 92% accuracy on validation data with a loss of ~0.62. Detailed metrics are available through the MLflow tracking interface.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed by [Khalid](https://github.com/khalidasdsju)

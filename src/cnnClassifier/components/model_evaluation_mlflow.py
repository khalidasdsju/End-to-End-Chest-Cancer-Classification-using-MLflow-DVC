import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.pyfunc
import os
import tempfile
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)


    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self):
        try:
            # Set MLFlow tracking URI
            print(f"Setting MLFlow tracking URI to: {self.config.mlflow_uri}")
            mlflow.set_tracking_uri(self.config.mlflow_uri)

            # Credentials will be automatically picked up from environment variables
            # MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
            import os as os_module
            print(f"MLFlow username: {os_module.environ.get('MLFLOW_TRACKING_USERNAME')}")
            print(f"MLFlow password is set: {bool(os_module.environ.get('MLFLOW_TRACKING_PASSWORD'))}")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(f"Tracking URL type: {tracking_url_type_store}")

            # Get experiment information for debugging
            print("Current experiment info:")
            try:
                current_experiment = mlflow.get_experiment_by_name("cnnclassifier")
                if current_experiment:
                    print(f"  - {current_experiment.name} (ID: {current_experiment.experiment_id})")
                else:
                    print("  - No experiment named 'cnnclassifier' found")
            except Exception as e:
                print(f"  - Error getting experiment info: {str(e)}")

            print("Starting MLFlow run...")
            with mlflow.start_run():
                print("Logging parameters...")
                mlflow.log_params(self.config.all_params)

                print("Logging metrics...")
                mlflow.log_metrics(
                    {"loss": self.score[0], "accuracy": self.score[1]}
                )

                print(f"Logging model (tracking type: {tracking_url_type_store})...")
                # Create a temporary file with the correct extension
                import tempfile
                import os as os_module

                # Save the model to a temporary file with .keras extension
                temp_dir = tempfile.mkdtemp()
                model_path = os_module.path.join(temp_dir, "model.keras")
                self.model.save(model_path)

                # Log the saved model to MLFlow and register it in the Model Registry
                model_name = "ChestCancerClassifierModel"

                if tracking_url_type_store != "file":
                    # Use pyfunc module to log the model

                    # Create a wrapper class for the model
                    class KerasModelWrapper(mlflow.pyfunc.PythonModel):
                        def __init__(self, model_path):
                            self.model_path = model_path

                        def load_context(self, context):
                            import tensorflow as tf
                            self.model = tf.keras.models.load_model(self.model_path)

                        def predict(self, context, model_input):
                            return self.model.predict(model_input)

                    # Log the model
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=KerasModelWrapper(model_path),
                        artifacts={"model_path": model_path},
                        registered_model_name=model_name
                    )
                    print(f"Model saved to {model_path} and registered in MLFlow Model Registry as '{model_name}'")

                    # Get the latest version of the registered model
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()

                    # Get the latest version
                    try:
                        latest_version = max([model.version for model in client.search_model_versions(f"name='{model_name}'")])
                        print(f"Latest version of the model: {latest_version}")

                        # Transition the model to 'Staging'
                        client.transition_model_version_stage(
                            name=model_name,
                            version=latest_version,
                            stage="Staging"
                        )
                        print(f"Model version {latest_version} transitioned to 'Staging' stage")
                    except Exception as e:
                        print(f"Warning: Could not transition model to staging: {str(e)}")
                else:
                    # For local tracking, just log the model without registering

                    # Create a wrapper class for the model
                    class KerasModelWrapper(mlflow.pyfunc.PythonModel):
                        def __init__(self, model_path):
                            self.model_path = model_path

                        def load_context(self, context):
                            import tensorflow as tf
                            self.model = tf.keras.models.load_model(self.model_path)

                        def predict(self, context, model_input):
                            return self.model.predict(model_input)

                    # Log the model
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=KerasModelWrapper(model_path),
                        artifacts={"model_path": model_path}
                    )
                    print(f"Model saved to {model_path} and logged to MLFlow (local tracking)")
                print("MLFlow run completed successfully")
        except Exception as e:
            print(f"Error in MLFlow logging: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

client.set_registered_model_alias("RandomForest", "champion", "1")
print("✅ Model promoted!")

from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from datetime import date
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from MLflowFunctions import load_mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# Load data
fs = feature_store.FeatureStoreClient()
query = "SELECT * FROM example.serving"
data = spark.sql(query)
model_name = "example_classifier"

# Load ML pipeline artifact from MLflow
clf = load_mlflow(model_name=model_name, stage="Production")

# Make predictions
todays_date = date.today()
data = data.toPandas()
preds = clf.predict_proba(data)
preds = [x[1] for x in preds]
data["Prediction"] = preds
data["Prediction_Date"] = [todays_date] * len(preds)

# Store predictions in Blob
path = f"/example_classifier_predictions/{todays_date}.csv"
data = data[["ID", "Prediction", "Preidction_Date"]]
data.to_csv(path)
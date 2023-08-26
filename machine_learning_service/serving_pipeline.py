from dataclasses import dataclass

from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from datetime import date
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from helperFunctions import load_mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from utils.constants import (
    COLS_FOR_REMOVAL,
    MODEL_NAME,
    MODEL_SERVING_QUERY,
    PREDICTION_COLS,
)
from utils.helperfunctions import load_mlflow


@dataclass
class ServingPipeline:
    def query_and_aggregate(self):
        """
        Queries and aggregates upstream data for inference

        :returns data: MRDS for inference
        """
        fs = feature_store.FeatureStoreClient()
        data = spark.sql(MODEL_SERVING_QUERY)
        data = data.toPandas()
        return data

    def serve_predictions(self):
        """
        Loads model artifact from MLflow serves predictions

        :returns None
        """
        # Get data for prediction
        data = self.query_and_aggregate()

        # Load production model artifact from Mlflow
        clf = load_mlflow(model_name=MODEL_NAME, stage="Production")

        # Serve predictions
        todays_date = date.today()

        # Make predictions
        todays_date = date.today()
        del data[COLS_FOR_REMOVAL]
        preds = clf.predict(data)
        data["Prediction"] = preds

        # Watermark predictions
        data["Prediction_Date"] = [todays_date] * len(preds)

        # Store predictions in Blob
        path = f"/example_classifier_predictions/{todays_date}.csv"
        data = data[PREDICTION_COLS]
        data.to_csv(path)

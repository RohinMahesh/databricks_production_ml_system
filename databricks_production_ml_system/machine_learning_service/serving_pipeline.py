from dataclasses import dataclass
from datetime import date

import pandas as pd
from databricks import feature_store
from databricks_production_ml_system.utils.constants import (
    COLS_FOR_REMOVAL,
    MODEL_NAME,
    MODEL_SERVING_QUERY,
    PREDICTION_COLS,
)
from databricks_production_ml_system.utils.helperFunctions import load_mlflow
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

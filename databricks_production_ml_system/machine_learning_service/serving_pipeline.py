from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from databricks import feature_store
from databricks_production_ml_system.utils.constants import (
    COLS_FOR_REMOVAL,
    MODEL_NAME,
    MODEL_SERVING_QUERY,
    PREDICTION_COLS,
    PREDICTION_DATE,
    TODAY,
)
from databricks_production_ml_system.utils.file_paths import PREDICTION_PATH
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
        todays_date = TODAY.strftime("%Y-%m-%d")
        del data[COLS_FOR_REMOVAL]
        preds = clf.predict(data)
        data["prediction"] = preds

        # Watermark predictions
        data[PREDICTION_DATE] = [todays_date] * len(preds)

        # Store predictions in Blob
        data = data[PREDICTION_COLS]
        data.write.mode("append").parquet(PREDICTION_PATH)

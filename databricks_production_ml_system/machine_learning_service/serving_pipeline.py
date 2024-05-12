from dataclasses import dataclass

import pandas as pd
from databricks import feature_store

from databricks_production_ml_system.utils.constants import (
    COLS_FOR_REMOVAL,
    MLFLOW_PROD_ENV,
    MODEL_NAME,
    MODEL_SERVING_QUERY,
    PREDICTION_COLS,
    PREDICTION_DATE,
    TODAY,
)
from databricks_production_ml_system.utils.file_paths import PREDICTION_PATH
from databricks_production_ml_system.utils.helpers import load_mlflow


@dataclass
class ServingPipeline:
    """
    ML pipeline for serving predictions
    """

    def _query_and_aggregate(self) -> pd.DataFrame:
        """
        Queries and aggregates upstream data for inference

        :returns data: MRDS for inference
        """
        fs = feature_store.FeatureStoreClient()
        data = spark.sql(MODEL_SERVING_QUERY)
        data = data.toPandas()
        return data

    def serve_predictions(self) -> None:
        """
        Loads model artifact from MLflow serves predictions

        :returns None
        """
        # Get data for prediction
        data = self._query_and_aggregate()

        # Load production model artifact from Mlflow
        clf = load_mlflow(model_name=MODEL_NAME, stage=MLFLOW_PROD_ENV)

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

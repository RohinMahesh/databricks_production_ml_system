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
from databricks_production_ml_system.utils.file_paths import PREDICTIONS_PATH
from databricks_production_ml_system.utils.helpers import check_data_exists, load_mlflow
from pyspark.sql import SparkSession


@dataclass
class ServingPipeline:
    """
    ML pipeline for serving predictions

    :param spark: SparkSession object
    """

    spark: SparkSession = None

    def _query_and_aggregate(self) -> pd.DataFrame:
        """
        Queries and aggregates upstream data for inference

        :returns data: MRDS for inference
        """
        fs = feature_store.FeatureStoreClient()
        data = (
            self.spark.sql(MODEL_SERVING_QUERY)
            if self.spark
            else spark.sql(MODEL_SERVING_QUERY)
        )
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
        data = data.drop(columns=COLS_FOR_REMOVAL)
        preds = clf.predict(data)
        data["prediction"] = preds

        # Watermark predictions
        data[PREDICTION_DATE] = [todays_date] * len(preds)

        # Store predictions in Blob
        data = data[PREDICTION_COLS]
        spark_df = (
            self.spark.createDataFrame(data)
            if self.spark
            else spark.createDataFrame(data)
        )
        exists = check_data_exists(f_path=PREDICTIONS_PATH)
        write_mode = "append" if exists else "overwrite"
        spark_df.write.mode(write_mode).parquet(PREDICTIONS_PATH)

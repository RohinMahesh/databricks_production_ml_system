from dataclasses import dataclass

import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from databricks import feature_store
from sklearn.metrics import f1_score

from databricks_production_ml_system.machine_learning_service.training_pipeline import (
    TrainingPipeline,
)
from databricks_production_ml_system.utils.constants import (
    CUTOFF_EVAL,
    DATA_FORMAT,
    DATE_COL,
    PERFORMANCE_EVAL_COLS,
    PERFORMANCE_EVAL_QUERY,
    PREDICTION_DATE,
    TARGET_COL,
    THRESHOLD_RETRAIN,
)
from databricks_production_ml_system.utils.file_paths import PREDICTIONS_PATH


@dataclass
class PerformanceEvaluation:
    """
    Evaluates performance and triggers model retraining based on SLA(s)

    :param spark: SparkSession object
    """

    spark: SparkSession = None

    def evaluate(self) -> None:
        """
        Evaluates model performance

        :returns None
        """
        # Get and filter predictions
        predictions = (
            self.spark.read.format(DATA_FORMAT).load(PREDICTIONS_PATH)
            if self.spark
            else spark.read.format(DATA_FORMAT).load(PREDICTIONS_PATH)
        )
        predictions = predictions.filter(
            func.col(PREDICTION_DATE) >= CUTOFF_EVAL
        ).withColumnRenamed(PREDICTION_DATE, DATE_COL)

        # Load labels
        fs = feature_store.FeatureStoreClient()
        data = (
            self.spark.sql(PERFORMANCE_EVAL_QUERY)
            if self.spark
            else spark.sql(PERFORMANCE_EVAL_QUERY)
        )
        data = data.select(PERFORMANCE_EVAL_COLS)
        to_evaluate = data.join(predictions, on=DATE_COL, how="inner").toPandas()

        # Calculate performance
        performance = float(
            f1_score(
                to_evaluate[TARGET_COL].tolist(), to_evaluate["prediction"].tolist()
            )
        )

        # Trigger retraining
        if performance < THRESHOLD_RETRAIN:
            TrainingPipeline().train_and_register_model()

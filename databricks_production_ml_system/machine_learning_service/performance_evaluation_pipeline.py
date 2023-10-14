from dataclasses import dataclass

import pandas as pd
import pyspark.sql.functions as func
from databricks import feature_store
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
    PREDICTIONS_PATH,
    THRESHOLD_RETRAIN,
)
from databricks_production_ml_system.utils.file_paths import PREDICTIONS_PATH
from sklearn.metrics import accuracy_score


@dataclass
class PerformanceEvaluation:
    """
    Evaluates performance and triggers model retraining based on SLA(s)
    """

    def evaluate(self):
        """
        Evaluates model performance
        """
        # Get and filter predictions
        predictions = spark.read.format(DATA_FORMAT).load(PREDICTIONS_PATH)
        predictions = predictions.filter(
            func.col(PREDICTION_DATE) >= CUTOFF_EVAL
        ).withColumnRenamed(PREDICTION_DATE, DATE_COL)

        # Load labels
        fs = feature_store.FeatureStoreClient()
        data = spark.sql(PERFORMANCE_EVAL_QUERY)
        data = data.select(PERFORMANCE_EVAL_COLS)
        to_evaluate = data.join(predictions, on=DATE_COL, how="left").toPandas()

        # Calculate performance
        performance = accuracy_score(
            to_evaluate["target"].tolist(), to_evaluate["prediction"].tolist()
        )

        # Trigger retraining
        if performance < THRESHOLD_RETRAIN:
            TrainingPipeline().train_and_register_model()

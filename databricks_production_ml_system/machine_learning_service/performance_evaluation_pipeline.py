from dataclasses import dataclass

import pandas as pd
from databricks import feature_store
from databricks_production_ml_system.machine_learning_service.training_pipeline import (
    TrainingPipeline,
)
from databricks_production_ml_system.utils.constants import (
    CUTOFF_EVAL,
    PERFORMANCE_EVAL_COLS,
    PERFORMANCE_EVAL_QUERY,
    PREDICTIONS_PATH,
    THRESHOLD_RETRAIN,
)
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
        predictions = spark.read.options(header=True).csv(PREDICTIONS_PATH)
        predictions = predictions.filter(predictions.Prediction_Date >= CUTOFF_EVAL)

        # Load labels
        fs = feature_store.FeatureStoreClient()
        data = spark.sql(PERFORMANCE_EVAL_QUERY)
        data = data.select(PERFORMANCE_EVAL_COLS)
        to_evaluate = data.join(
            predictions, data.ID == predictions.ID, "left"
        ).toPandas()

        # Calculate performance
        performance = accuracy_score(
            to_evaluate["Target"].tolist(), to_evaluate["Prediction"].tolist()
        )

        # Trigger retraining
        if performance < THRESHOLD_RETRAIN:
            TrainingPipeline().train_and_register_model()

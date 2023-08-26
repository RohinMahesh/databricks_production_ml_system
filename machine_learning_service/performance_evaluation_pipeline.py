from dataclasses import dataclass

from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import pyspark.sql.functions as func
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils.constants import (
    CUTOFF_EVAL,
    PERFORMANCE_EVAL_COLS,
    PERFORMANCE_EVAL_QUERY,
    PREDICTIONS_PATH,
    THRESHOLD_RETRAIN,
)

from utils.helperfunctions import register_mlflow
from training_pipeline import TrainingPipeline


@dataclass
class PerformanceEvaluation:
    """
    Evaluates performance and triggers model retraining based on SLA(s)
    """

    def evaluate(self):
        """
        Evaluates model performance

        :returns None
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

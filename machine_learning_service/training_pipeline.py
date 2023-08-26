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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from utils.constants import (
    CATEGORICAL_COLUMNS,
    HYPERPARAMS,
    MODEL_TRAINING_QUERY,
    TARGET_COL,
)

from utils.helperfunctions import register_mlflow


@dataclass
class TrainingPipeline:
    """
    Trains and serializes ML Pipeline
    """

    def query_and_aggregate(self):
        """
        Queries and aggregates upstream data for training

        :returns data, feature_space: feature space and target variable
        """
        # Query data from offline feature table
        fs = feature_store.FeatureStoreClient()
        data = spark.sql(MODEL_TRAINING_QUERY)

        # Engineer features
        feature_space = (
            data.groupby("CustomerNumber")
            .agg(
                func.min("Feature1").alias("Feature1"),
                func.max("Feature2").alias("Feature2"),
                func.sum("Feature3").alias("Feature3"),
                func.last("Feature4").alias("Feature4"),
                func.last(TARGET_COL).alias(TARGET_COL),
            )
            .toPandas()
        )

        # Split data
        target = feature_space[TARGET_COL]
        del feature_space[TARGET_COL]
        return data, feature_space

    def train_and_register_model(self):
        """
        Trains ML model and serializes pipeline in MLflow
        """
        # Get feature space and target variable
        feature_space, target = self.query_and_aggregate()

        # Define feature extractor
        categorical_transformer = Pipeline(
            steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
        )
        feature_extractor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
            ],
            remainder="passthrough",
        )

        # Fit model
        clf = Pipeline(
            steps=[
                ("preprocessor", feature_extractor),
                ("model", LogisticRegression(**HYPERPARAMS)),
            ]
        )
        clf.fit(feature_space, target)

        # Register model artifact in MLflow
        register_mlflow(
            data=feature_space,
            model=clf,
        )

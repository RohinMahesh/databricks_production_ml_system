from dataclasses import dataclass

import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from databricks import feature_store
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from databricks_production_ml_system.utils.constants import (
    CATEGORICAL_COLS,
    HYPERPARAMS,
    MODEL_TRAINING_QUERY,
    TARGET_COL,
)


@dataclass
class TrainingPipeline:
    """
    Trains and serializes ML Pipeline

    :param spark: SparkSession object
    """

    spark: SparkSession = None

    def _query_and_aggregate(self) -> None:
        """
        Queries and aggregates upstream data for training

        :returns None
        """
        # Query data from offline feature table
        fs = feature_store.FeatureStoreClient()
        data = (
            self.spark.sql(MODEL_TRAINING_QUERY)
            if self.spark
            else spark.sql(MODEL_TRAINING_QUERY)
        )

        # Engineer features
        self.feature_space = (
            data.groupby("customer_number")
            .agg(
                func.min("feature1").alias("feature1"),
                func.max("feature2").alias("feature2"),
                func.sum("feature3").alias("feature3"),
                func.last("feature4").alias("feature4"),
                func.last(TARGET_COL).alias(TARGET_COL),
            )
            .toPandas()
        )

        # Split data
        self.target = self.feature_space[TARGET_COL]
        del self.feature_space[TARGET_COL]

    def train_and_register_model(self) -> None:
        """
        Trains ML model and serializes pipeline in MLflow

        :returns None
        """
        from databricks_production_ml_system.utils.helpers import register_mlflow

        # Get feature space and target variable
        self._query_and_aggregate()

        # Define feature extractor
        categorical_transformer = Pipeline(
            steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
        )
        feature_extractor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, CATEGORICAL_COLS),
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
        clf.fit(self.feature_space, self.target)

        # Register model artifact in MLflow
        register_mlflow(data=self.feature_space, model=clf)

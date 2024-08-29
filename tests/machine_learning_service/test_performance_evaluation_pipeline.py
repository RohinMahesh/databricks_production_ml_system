import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from databricks_production_ml_system.machine_learning_service.performance_evaluation_pipeline import (
    PerformanceEvaluation,
)
from databricks_production_ml_system.utils.constants import (
    DATE_COL,
    PREDICTION_DATE,
    TARGET_COL,
)
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def unittest_spark():
    return (
        SparkSession.builder.appName("unit-test-helpers")
        .master("local[*]")
        .getOrCreate()
    )


@pytest.fixture
def mock_data(request, unittest_spark):
    if request.param == "low_performance":
        predictions = pd.DataFrame(
            {PREDICTION_DATE: ["2024-08-28", "2024-08-29"], "prediction": [0, 0]}
        )
        data = pd.DataFrame(
            {DATE_COL: ["2024-08-28", "2024-08-29"], TARGET_COL: [1, 1]}
        )
    else:
        predictions = pd.DataFrame(
            {PREDICTION_DATE: ["2024-08-28", "2024-08-29"], "prediction": [1, 0]}
        )
        data = pd.DataFrame(
            {DATE_COL: ["2024-08-28", "2024-08-29"], TARGET_COL: [1, 0]}
        )

    predictions_df = unittest_spark.createDataFrame(predictions)
    data_df = unittest_spark.createDataFrame(data)

    return {"predictions": predictions_df, "data": data_df}


@pytest.mark.parametrize(
    "mock_data, should_retrain",
    [
        ("low_performance", True),
        ("high_performance", False),
    ],
    indirect=["mock_data"],
)
def test_evaluate(mock_data, unittest_spark, should_retrain):
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_data["predictions"].write.parquet(f"{temp_dir}/predictions.parquet")

        with patch(
            "databricks_production_ml_system.machine_learning_service.performance_evaluation_pipeline.feature_store.FeatureStoreClient"
        ), patch(
            "databricks_production_ml_system.machine_learning_service.performance_evaluation_pipeline.TrainingPipeline"
        ) as MockTrainingPipeline, patch(
            "databricks_production_ml_system.machine_learning_service.performance_evaluation_pipeline.PREDICTIONS_PATH",
            f"{temp_dir}/predictions.parquet",
        ), patch.object(
            unittest_spark.read,
            "format",
            return_value=MagicMock(
                load=MagicMock(return_value=mock_data["predictions"])
            ),
        ), patch.object(
            unittest_spark, "sql", return_value=mock_data["data"]
        ):

            evaluator = PerformanceEvaluation(spark=unittest_spark)
            evaluator.evaluate()

            if should_retrain:
                MockTrainingPipeline.assert_called_once()
            else:
                MockTrainingPipeline.assert_not_called()

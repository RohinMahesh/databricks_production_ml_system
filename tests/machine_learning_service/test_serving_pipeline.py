import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from databricks_production_ml_system.machine_learning_service.serving_pipeline import (
    ServingPipeline,
)
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType
from sklearn.linear_model import LogisticRegression


def test_query_and_aggregate():
    unittest_spark = (
        SparkSession.builder.appName("unit-test-query-and-aggregate")
        .master("local[*]")
        .getOrCreate()
    )

    data_list = [(1, 0.1, 1), (2, 0.2, 0), (3, 0.3, 1)]

    schema = StructType(
        [
            StructField("customer_number", IntegerType(), True),
            StructField("feature1", FloatType(), True),
            StructField("feature2", IntegerType(), True),
        ]
    )

    mock_df = unittest_spark.createDataFrame(data_list, schema)

    with patch(
        "databricks_production_ml_system.machine_learning_service.serving_pipeline.SparkSession.sql",
        return_value=mock_df,
    ):
        with patch(
            "databricks_production_ml_system.machine_learning_service.serving_pipeline.feature_store.FeatureStoreClient",
            MagicMock(),
        ):
            pipeline = ServingPipeline(spark=unittest_spark)

            result = pipeline._query_and_aggregate()
            result_dict = result.to_dict(orient="list")
            result_dict["feature1"] = [round(x, 1) for x in result_dict["feature1"]]

            expected_dict = {
                "customer_number": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [1, 0, 1],
            }
            assert (
                result_dict == expected_dict
            ), f"Expected {expected_dict}, but got {result_dict}"


@pytest.mark.parametrize("data_exists", [True, False])
def test_serve_predictions(data_exists):
    unittest_spark = (
        SparkSession.builder.appName("unit-test-serving-pipeline")
        .master("local[*]")
        .getOrCreate()
    )
    data_list = [(1, 0.1, 1), (2, 0.2, 0), (3, 0.3, 1)]

    schema = StructType(
        [
            StructField("customer_number", IntegerType(), True),
            StructField("feature1", FloatType(), True),
            StructField("feature2", IntegerType(), True),
        ]
    )

    mock_df = unittest_spark.createDataFrame(data_list, schema)

    with patch(
        "databricks_production_ml_system.machine_learning_service.serving_pipeline.SparkSession.sql",
        return_value=mock_df,
    ):
        with patch(
            "databricks_production_ml_system.machine_learning_service.serving_pipeline.feature_store.FeatureStoreClient",
            MagicMock(),
        ):
            pipeline = ServingPipeline(spark=unittest_spark)

            X = pd.DataFrame({"feature1": [0.1, 0.2, 0.3], "feature2": [1, 0, 1]})
            y = [1, 0, 1]
            clf = LogisticRegression()
            clf.fit(X, y)

            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = os.path.join(tmp_dir, "model.pkl")

                with open(model_path, "wb") as model_file:
                    pickle.dump(clf, model_file)

                with patch(
                    "databricks_production_ml_system.machine_learning_service.serving_pipeline.load_mlflow",
                    return_value=pickle.load(open(model_path, "rb")),
                ):
                    with patch(
                        "databricks_production_ml_system.machine_learning_service.serving_pipeline.check_data_exists",
                        return_value=data_exists,
                    ):
                        with patch(
                            "databricks_production_ml_system.machine_learning_service.serving_pipeline.PREDICTIONS_PATH",
                            tmp_dir,
                        ):
                            pipeline.serve_predictions()

                            files = [
                                f for f in os.listdir(tmp_dir) if f.endswith(".parquet")
                            ]
                            assert len(files) > 0, "No parquet files were written!"
                            assert os.path.exists(os.path.join(tmp_dir, files[0]))

from unittest import mock

import pytest
from databricks_production_ml_system.machine_learning_service.training_pipeline import (
    TrainingPipeline,
)
from databricks_production_ml_system.utils.constants import TARGET_COL
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


@pytest.mark.parametrize(
    "mock_data, expected_feature_space, expected_target",
    [
        (
            [
                ("12345", 10, 20, 30, 40, 1),
                ("12345", 15, 25, 35, 45, 1),
                ("67890", 5, 15, 25, 35, 0),
            ],
            {
                "customer_number": {0: "12345", 1: "67890"},
                "feature1": {0: 10, 1: 5},
                "feature2": {0: 25, 1: 15},
                "feature3": {0: 65, 1: 25},
                "feature4": {0: 45, 1: 35},
            },
            {0: 1, 1: 0},
        ),
        (
            [
                ("11111", 1, 2, 3, 4, 1),
                ("11111", 2, 3, 4, 5, 0),
                ("22222", 3, 4, 5, 6, 1),
            ],
            {
                "customer_number": {0: "11111", 1: "22222"},
                "feature1": {0: 1, 1: 3},
                "feature2": {0: 3, 1: 4},
                "feature3": {0: 7, 1: 5},
                "feature4": {0: 5, 1: 6},
            },
            {0: 0, 1: 1},
        ),
    ],
)
def test_query_and_aggregate(mock_data, expected_feature_space, expected_target):
    unittest_spark = (
        SparkSession.builder.appName("unit-test-training-pipeline")
        .master("local[*]")
        .getOrCreate()
    )
    schema = StructType(
        [
            StructField("customer_number", StringType(), True),
            StructField("feature1", IntegerType(), True),
            StructField("feature2", IntegerType(), True),
            StructField("feature3", IntegerType(), True),
            StructField("feature4", IntegerType(), True),
            StructField(TARGET_COL, IntegerType(), True),
        ]
    )

    df = unittest_spark.createDataFrame(mock_data, schema=schema)

    pipeline = TrainingPipeline(spark=unittest_spark)

    with mock.patch.object(unittest_spark, "sql", return_value=df):
        pipeline._query_and_aggregate()

    feature_space = pipeline.feature_space.to_dict()
    target = pipeline.target.to_dict()

    assert feature_space == expected_feature_space
    assert target == expected_target


@pytest.mark.parametrize(
    "mock_data",
    [
        [
            ("12345", 10, 20, 30, 40, 1),
            ("12345", 15, 25, 35, 45, 1),
            ("67890", 5, 15, 25, 35, 0),
        ],
        [
            ("11111", 1, 2, 3, 4, 1),
            ("11111", 2, 3, 4, 5, 0),
            ("22222", 3, 4, 5, 6, 1),
        ],
    ],
)
@mock.patch("databricks_production_ml_system.utils.helpers.register_mlflow")
def test_train_and_register_model(mock_register_mlflow, mock_data):
    unittest_spark = (
        SparkSession.builder.appName("unit-test-training-pipeline")
        .master("local[*]")
        .getOrCreate()
    )
    schema = StructType(
        [
            StructField("customer_number", StringType(), True),
            StructField("feature1", IntegerType(), True),
            StructField("feature2", IntegerType(), True),
            StructField("feature3", IntegerType(), True),
            StructField("feature4", IntegerType(), True),
            StructField(TARGET_COL, IntegerType(), True),
        ]
    )

    df = unittest_spark.createDataFrame(mock_data, schema=schema)

    pipeline = TrainingPipeline(spark=unittest_spark)

    with mock.patch.object(unittest_spark, "sql", return_value=df):
        pipeline.train_and_register_model()

    mock_register_mlflow.assert_called_once()

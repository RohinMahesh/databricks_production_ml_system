import os
import tempfile
from unittest import mock

import pytest
from databricks_production_ml_system.data_engineering_service.online_feature_table_daily import (
    feature_store_online_serving_update,
)
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


@pytest.mark.parametrize(
    "mock_data, expected_count",
    [
        (
            [
                ("12345", 10, 20, 30, 40),
                ("12345", 15, 25, 35, 45),
                ("67890", 5, 15, 25, 35),
            ],
            2,
        ),
        (
            [
                ("11111", 1, 2, 3, 4),
                ("11111", 2, 3, 4, 5),
                ("22222", 3, 4, 5, 6),
            ],
            2,
        ),
    ],
)
def test_feature_store_online_serving_update(mock_data, expected_count):
    unittest_spark = (
        SparkSession.builder.appName("unit-test-feature-store-online-serving")
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
        ]
    )

    df = unittest_spark.createDataFrame(mock_data, schema=schema)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "mock_output.parquet")

        def mock_update_table(*args, **kwargs):
            data = kwargs.get("data")
            data.write.parquet(temp_path)

        with mock.patch.object(unittest_spark, "sql", return_value=df), mock.patch(
            "databricks_production_ml_system.data_engineering_service.online_feature_table_daily.update_table",
            side_effect=mock_update_table,
        ) as mock_update_table_func:
            feature_store_online_serving_update(spark=unittest_spark)

            mock_update_table_func.assert_called_once()
            assert os.path.exists(
                temp_path
            ), f"Expected file at {temp_path} does not exist."

            saved_df = unittest_spark.read.parquet(temp_path)
            assert saved_df.count() == expected_count

            parquet_files = [f for f in os.listdir(temp_dir) if f.endswith(".parquet")]
            assert (
                len(parquet_files) > 0
            ), "No parquet files were found in the temporary directory"

import os
import sys
from unittest import mock

sys.modules["dlt"] = mock.Mock()

import tempfile

import pytest
from databricks_production_ml_system.data_engineering_service.feature_store_training_daily import (
    feature_store_offline_training_update,
)
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


@pytest.mark.parametrize(
    "mock_data, expected_call",
    [
        ([], False),
        (
            [("12345", 1, 2, 3, 4, "2024-08-01", 0)],
            True,
        ),
    ],
)
def test_feature_store_offline_training_update(mock_data, expected_call):
    unittest_spark = (
        SparkSession.builder.appName("unit-test-feature-store-training-daily")
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
            StructField("date", StringType(), True),
            StructField("target", IntegerType(), True),
        ]
    )

    df = unittest_spark.createDataFrame(mock_data, schema=schema)

    mock_dlt = sys.modules["dlt"]
    mock_dlt.load.return_value = df

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "mock_output.parquet")

        def mock_update_table(*args, **kwargs):
            data = kwargs.get("data")
            if expected_call:
                data.write.parquet(temp_path)

        with mock.patch(
            "databricks_production_ml_system.data_engineering_service.feature_store_training_daily.update_table",
            side_effect=mock_update_table,
        ) as mock_update_table_func:
            with mock.patch(
                "databricks_production_ml_system.utils.helpers.feature_store"
            ) as mock_fs:
                mock_fs.create_feature_table.return_value = None
                mock_fs._compute_client._spark_client_helper.check_catalog_database_exists.return_value = (
                    True
                )

                feature_store_offline_training_update(spark=unittest_spark)

                if expected_call:
                    mock_update_table_func.assert_called_once()
                    assert os.path.exists(
                        temp_path
                    ), f"Expected file at {temp_path} does not exist."

                    saved_df = unittest_spark.read.parquet(temp_path)
                    assert saved_df.count() == len(mock_data)

                    parquet_files = [
                        f for f in os.listdir(temp_dir) if f.endswith(".parquet")
                    ]
                    assert (
                        len(parquet_files) > 0
                    ), "No parquet files were found in the temporary directory"
                else:
                    mock_update_table_func.assert_not_called()

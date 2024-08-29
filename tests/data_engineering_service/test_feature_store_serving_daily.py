import sys
from unittest import mock

sys.modules["dlt"] = mock.Mock()

import os
import tempfile

import pytest
from databricks_production_ml_system.data_engineering_service.feature_store_serving_daily import (
    feature_store_offline_serving_update,
)
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.sql.utils import AnalysisException

schema = StructType(
    [StructField("id", IntegerType(), True), StructField("value", StringType(), True)]
)

data = [(1, "A"), (2, "B")]


@pytest.mark.parametrize("data_exists", [True, False])
@mock.patch("databricks.feature_store.FeatureStoreClient", autospec=True)
@mock.patch(
    "databricks_production_ml_system.utils.helpers.publish_table", autospec=True
)
@mock.patch("databricks_production_ml_system.utils.helpers.update_table", autospec=True)
def test_feature_store_offline_serving_update(
    mock_update_table, mock_publish_table, mock_fs_client, data_exists
):
    unittest_spark = (
        SparkSession.builder.appName("unit-test-feature-store-serving-daily")
        .master("local[*]")
        .getOrCreate()
    )

    mock_dlt = sys.modules["dlt"]
    mock_dlt.load.return_value = unittest_spark.createDataFrame(data, schema=schema)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_parquet_path = os.path.join(temp_dir, "sample_data.parquet")
        unittest_spark.createDataFrame(data, schema=schema).write.parquet(
            temp_parquet_path
        )

        if data_exists:
            mock_fs_client.return_value.read_table.return_value = (
                unittest_spark.read.parquet(temp_parquet_path)
            )
        else:
            mock_fs_client.return_value.read_table.side_effect = AnalysisException(
                message="Table or view not found",
                error_class=None,
                message_parameters=None,
            )

        feature_store_offline_serving_update(spark=unittest_spark)

        files_in_temp_dir = os.listdir(temp_dir)
        assert (
            len(files_in_temp_dir) > 0
        ), "No files were found in the temp directory after function execution."

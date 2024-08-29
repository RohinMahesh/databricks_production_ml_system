import os
import shutil
from datetime import datetime
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
from databricks_production_ml_system.utils.constants import DATE_COL, TARGET_COL
from databricks_production_ml_system.utils.file_paths import RAW_FILE_PATH
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

unittest_spark = (
    SparkSession.builder.appName("unit-test-create-data")
    .master("local[*]")
    .getOrCreate()
)


def clear_dir(dir_path: str) -> None:
    """
    Clears directory given path

    :param dir_path: path to be cleared
    :returns None
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


@pytest.mark.parametrize("sample_size, data_exists", [(10, False), (100, True)])
@mock.patch("databricks_production_ml_system.utils.helpers.check_data_exists")
def test_generate_random_data(mock_check_data_exists, sample_size, data_exists):
    mock_check_data_exists.return_value = data_exists
    schema = StructType(
        [
            StructField("customer_number", IntegerType(), nullable=False),
            StructField("feature1", DoubleType(), nullable=True),
            StructField("feature2", DoubleType(), nullable=True),
            StructField("feature3", DoubleType(), nullable=True),
            StructField("feature4", StringType(), nullable=True),
            StructField(DATE_COL, DateType(), nullable=False),
            StructField(TARGET_COL, IntegerType(), nullable=False),
        ]
    )
    with TemporaryDirectory() as tmpdir:
        with mock.patch(
            "databricks_production_ml_system.utils.file_paths.RAW_FILE_PATH", tmpdir
        ):
            from databricks_production_ml_system.data_engineering_service.create_data import (
                generate_random_data,
            )

            if data_exists:
                dummy_data = unittest_spark.createDataFrame(
                    [
                        (1, 10.0, 20.0, 30.0, "Low", datetime.now(), 0),
                        (2, 15.0, 25.0, 35.0, "Medium", datetime.now(), 1),
                    ],
                    schema=schema,
                )
                dummy_data.write.mode("overwrite").parquet(tmpdir)
                files_in_dir = os.listdir(tmpdir)
                assert len(files_in_dir) > 0, "Pre-existing data not written correctly."

            generate_random_data(sample_size=sample_size, spark=unittest_spark)

            written_files = unittest_spark.read.parquet(tmpdir, schema=schema)

            assert written_files.count() > 0, "No records found in directory"

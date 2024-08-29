from datetime import datetime

import numpy as np
import pyspark.sql.functions as func
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from databricks_production_ml_system.utils.constants import DATE_COL, TARGET_COL
from databricks_production_ml_system.utils.file_paths import RAW_FILE_PATH
from databricks_production_ml_system.utils.helpers import check_data_exists


def generate_random_data(sample_size: int = 10000, spark: SparkSession = None) -> None:
    """
    Generates and saves random data of given sample size

    :param sample_size: number of rows in randomly generated data
    :param spark: SparkSession object
    :returns None
    """
    # Define schema
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

    # Randomly generate data
    data = {
        "customer_number": np.random.randint(
            low=1, high=100, size=sample_size
        ).tolist(),
        "feature1": np.random.uniform(low=0, high=75, size=sample_size).tolist(),
        "feature2": np.random.uniform(low=0, high=30, size=sample_size).tolist(),
        "feature3": np.random.normal(loc=3, scale=5, size=sample_size).tolist(),
        "feature4": np.random.choice(
            a=["Low", "Medium", "High", "Unknown"],
            size=sample_size,
            p=[0.33, 0.33, 0.33, 0.01],
        ).tolist(),
        DATE_COL: [datetime.now().strftime("%Y-%m-%d")] * sample_size,
        TARGET_COL: np.random.choice(a=[0, 1], size=sample_size, p=[0.4, 0.6]).tolist(),
    }
    data[DATE_COL] = [datetime.strptime(x, "%Y-%m-%d") for x in data[DATE_COL]]
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    data = (
        spark.createDataFrame(data, schema)
        .withColumn("feature1", func.round("feature1", 2))
        .withColumn("feature2", func.round("feature2", 2))
        .withColumn("feature3", func.round("feature3", 2))
    )

    # Save data
    exists = check_data_exists(f_path=RAW_FILE_PATH)
    write_mode = "append" if exists else "overwrite"
    import logging

    logging.info(f"directory: {RAW_FILE_PATH}")
    data.write.mode(write_mode).parquet(RAW_FILE_PATH)

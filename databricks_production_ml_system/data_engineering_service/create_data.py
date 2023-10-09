from datetime import datetime

import numpy as np
import pyspark.sql.functions as func
from databricks_production_ml_system.utils.constants import (
    DATE_COL,
    FILEPATH,
    TARGET_COL,
)
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def generate_random_data(sample_size: str, incremental: True):
    """
    Generates and saves random data of given sample size

    :param sample_size: number of rows in randomly generated data
    :param incremental: whether the data is incremental
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
            StructField(DATE_COL, TimestampType(), nullable=False),
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
        "feature3": np.random.normal(loc=3, scale==5, size=sample_size).tolist(),
        "feature4": np.random.choice(
            a=["Low", "Medium", "High", "Unknown"],
            size=sample_size,
            p=[0.33, 0.33, 0.33, 0.01],
        ).tolist(),
        DATE_COL: [datetime.now().strftime("%Y-%m-%d")] * sample_size,
        TARGET_COL: np.random.choice(a=[0, 1], size=sample_size, p=[0.4, 0.6]).tolist(),
    }
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    data = (
        spark.createDataFrame(data, schema)
        .withColumn("feature1", func.round("feature1", 2))
        .withColumn("feature2", func.round("feature2", 2))
        .withColumn("feature3", func.round("feature3", 2))
    )
    if incremental:
        data.write.mode("append").parquet(FILEPATH)
    else:
        return data
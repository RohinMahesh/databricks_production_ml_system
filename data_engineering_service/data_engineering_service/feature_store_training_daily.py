from databricks import feature_store
from datetime import datetime, timedelta
from pyspark.sql.functions import func
from pyspark.sql.window import Window
from pathlib import Path
import os
import random

from utils.constants import (
    CUTOFF,
    DATE_COL,
    DELIMITER,
    FILEPATH,
    OFFLINE_TABLE_DESCRIPTION_TRAINING,
    OFFLINE_TABLE_KEYS,
    OFFLINE_TABLE_PARTITION,
    OFFLINE_TABLE_SCHEMA,
    OFFLINE_TABLE_TRAINING,
    OFFLINE_TABLE_TRAINING_COLS,
    OFFINE_TABLE_TRAINING_DESCRIPTION,
)
from utils.helperfunctions import update_table

# Load in table
data = (
    spark.read.options(delimiter=DELIMITER, header=True)
    .csv(FILEPATH)
    .withColumn(
        "row_number",
        func.row_number().over(
            Window.partitionBy(OFFLINE_TABLE_PARTITION).orderBy(
                func.col(DATE_COL).desc()
            )
        ),
    )
    .where(func.col("row_number") == 1)
    .drop("row_number")
)
data = data.filter(func.col(DATE_COL) >= CUTOFF)

# Insert new records if new labeled data is available
if len(data.head(1)) > 0:
    # Select columns for downstream consumption
    data = data.select(OFFLINE_TABLE_TRAINING_COLS)
    # Impute odd level of Feature4
    data = data.withColumn(
        "Feature4",
        func.when(data.Feature4 == "Unknown", "Status1").otherwise(data.Feature4),
    )

    # Update feature table
    update_table(
        data=data,
        description=OFFLINE_TABLE_DESCRIPTION_TRAINING,
        schema=OFFLINE_TABLE_SCHEMA,
        table=OFFLINE_TABLE_TRAINING,
        keys=OFFLINE_TABLE_KEYS,
        partition_columns=OFFLINE_TABLE_PARTITION,
    )

import dlt
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.sql.window import Window

from databricks_production_ml_system.utils.constants import (
    ARDS_TABLE_NAME,
    DATE_COL,
    OFFLINE_TABLE_DESCRIPTION_TRAINING,
    OFFLINE_TABLE_KEYS,
    OFFLINE_TABLE_PARTITION,
    OFFLINE_TABLE_SCHEMA,
    OFFLINE_TABLE_TRAINING,
    OFFLINE_TABLE_TRAINING_COLS,
    ROW_NUMBER_COLUMN,
)
from databricks_production_ml_system.utils.helpers import update_table


def feature_store_offline_training_update(spark: SparkSession = None) -> None:
    """
    Executes daily update of the offline feature store for training

    :param spark: SparkSession object
    :returns None
    """
    # Load table
    data = (
        dlt.load(ARDS_TABLE_NAME)
        .withColumn(
            ROW_NUMBER_COLUMN,
            func.row_number().over(
                Window.partitionBy(OFFLINE_TABLE_PARTITION).orderBy(
                    func.col(DATE_COL).desc()
                )
            ),
        )
        .where(func.col(ROW_NUMBER_COLUMN) == 1)
        .drop(ROW_NUMBER_COLUMN)
    )

    # Insert new records if new labeled data is available
    if data.limit(1).count() > 0:
        # Select columns for downstream consumption
        data = data.select(OFFLINE_TABLE_TRAINING_COLS)

        update_table_args = {
            "data": data,
            "description": OFFLINE_TABLE_DESCRIPTION_TRAINING,
            "schema": OFFLINE_TABLE_SCHEMA,
            "table": OFFLINE_TABLE_TRAINING,
            "keys": OFFLINE_TABLE_KEYS,
            "partition_columns": OFFLINE_TABLE_PARTITION,
        }
        if spark:
            update_table_args["spark"] = spark

        # Update feature table
        update_table(**update_table_args)

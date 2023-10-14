import dlt
from databricks_production_ml_system.utils.constants import (
    ARDS_TABLE_NAME,
    CUTOFF,
    DATE_COL,
    OFFINE_TABLE_TRAINING_DESCRIPTION,
    OFFLINE_TABLE_DESCRIPTION_TRAINING,
    OFFLINE_TABLE_KEYS,
    OFFLINE_TABLE_PARTITION,
    OFFLINE_TABLE_SCHEMA,
    OFFLINE_TABLE_TRAINING,
    OFFLINE_TABLE_TRAINING_COLS,
)
from databricks_production_ml_system.utils.file_paths import RAW_FILE_PATH
from databricks_production_ml_system.utils.helperfunctions import update_table
from pyspark.sql.functions import func
from pyspark.sql.window import Window


def feature_store_offline_training_update():
    """
    Executes daily update of the offline feature store for training
    """
    # Load table
    data = (
        dlt.load(ARDS_TABLE_NAME)
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

    # Insert new records if new labeled data is available
    if len(data.head(1)) > 0:
        # Select columns for downstream consumption
        data = data.select(OFFLINE_TABLE_TRAINING_COLS)

        # Update feature table
        update_table(
            data=data,
            description=OFFLINE_TABLE_DESCRIPTION_TRAINING,
            schema=OFFLINE_TABLE_SCHEMA,
            table=OFFLINE_TABLE_TRAINING,
            keys=OFFLINE_TABLE_KEYS,
            partition_columns=OFFLINE_TABLE_PARTITION,
        )

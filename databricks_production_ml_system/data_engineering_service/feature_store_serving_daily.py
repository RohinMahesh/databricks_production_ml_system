from databricks_production_ml_system.utils.constants import (
    FILEPATH,
    OFFLINE_TABLE_DESCRIPTION_SERVING,
    OFFLINE_TABLE_KEYS,
    OFFLINE_TABLE_PARTITION,
    OFFLINE_TABLE_SCHEMA,
    OFFLINE_TABLE_SERVING,
)
from databricks_production_ml_system.utils.helperfunctions import (
    publish_table,
    update_table,
)


def feature_store_offline_serving_update():
    """
    Executes daily update of the offline feature table for serving
    """
    # Get latest CSV file
    data = spark.read.format("parquet").load(FILEPATH)

    # Update feature table
    update_table(
        data=data,
        description=OFFLINE_TABLE_DESCRIPTION_SERVING,
        schema=OFFLINE_TABLE_SCHEMA,
        table=OFFLINE_TABLE_SERVING,
        keys=OFFLINE_TABLE_KEYS,
        partition_columns=OFFLINE_TABLE_PARTITION,
    )
import dlt
from databricks import feature_store
from pyspark.sql.utils import AnalysisException

from databricks_production_ml_system.utils.constants import (
    ARDS_TABLE_NAME,
    OFFLINE_TABLE_DESCRIPTION_SERVING,
    OFFLINE_TABLE_KEYS,
    OFFLINE_TABLE_PARTITION,
    OFFLINE_TABLE_SCHEMA,
    OFFLINE_TABLE_SERVING,
)
from databricks_production_ml_system.utils.helpers import publish_table, update_table
from pyspark.sql import SparkSession


def feature_store_offline_serving_update(spark: SparkSession = None) -> None:
    """
    Executes daily update of the offline feature table for serving

    :param spark: SparkSession object
    :returns None
    """

    # Initialize Feature Store client
    fs = feature_store.FeatureStoreClient()

    try:
        # Read table if it does exist
        existing_table = fs.read_table(OFFLINE_TABLE_SERVING)
        update_table_args = {
            "data": existing_table,
            "description": OFFLINE_TABLE_DESCRIPTION_SERVING,
            "schema": OFFLINE_TABLE_SCHEMA,
            "table": OFFLINE_TABLE_SERVING,
            "keys": OFFLINE_TABLE_KEYS,
            "partition_columns": OFFLINE_TABLE_PARTITION,
        }
        if spark:
            update_table_args["spark"] = spark

        # Update feature table
        update_table(**update_table_args)

    except AnalysisException:
        # Load ARDS
        data = dlt.load(ARDS_TABLE_NAME)

        # If table does not exist, create it
        publish_table(
            schema=OFFLINE_TABLE_SCHEMA,
            table=OFFLINE_TABLE_SERVING,
        )

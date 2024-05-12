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


def feature_store_offline_serving_update() -> None:
    """
    Executes daily update of the offline feature table for serving

    :returns None
    """
    # Load ARDS
    data = dlt.load(ARDS_TABLE_NAME)

    # Initialize Feature Store client
    fs = feature_store.FeatureStoreClient()

    try:
        # Read table if it does exist
        existing_table = fs.read_table(OFFLINE_TABLE_SERVING)

        # Update feature table
        update_table(
            data=data,
            description=OFFLINE_TABLE_DESCRIPTION_SERVING,
            schema=OFFLINE_TABLE_SCHEMA,
            table=OFFLINE_TABLE_SERVING,
            keys=OFFLINE_TABLE_KEYS,
            partition_columns=OFFLINE_TABLE_PARTITION,
        )
    except AnalysisException:
        # If table does not exist, create it
        publish_table(
            data=data,
            description=OFFLINE_TABLE_DESCRIPTION_SERVING,
            schema=OFFLINE_TABLE_SCHEMA,
            table_name=OFFLINE_TABLE_SERVING,
            keys=OFFLINE_TABLE_KEYS,
            partition_columns=OFFLINE_TABLE_PARTITION,
        )

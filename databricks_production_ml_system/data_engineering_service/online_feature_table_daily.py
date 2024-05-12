from pyspark.sql.functions import func

from databricks_production_ml_system.utils.constants import (
    ONLINE_TABLE,
    ONLINE_TABLE_DESCRIPTION,
    ONLINE_TABLE_KEYS,
    ONLINE_TABLE_PARTITION,
    ONLINE_TABLE_QUERY,
    ONLINE_TABLE_SCHEMA,
)
from databricks_production_ml_system.utils.helpers import update_table


def feature_store_online_serving_update() -> None:
    """
    Executes daily update of the online feature table for serving

    :returns None
    """
    # Query data
    data = spark.sql(ONLINE_TABLE_QUERY)
    online_features = data.groupby("customer_number").agg(
        func.min("feature1").alias("feature1"),
        func.max("feature2").alias("feature2"),
        func.sum("feature3").alias("feature3"),
        func.last("feature4").alias("feature4"),
    )

    # Update feature table
    update_table(
        data=online_features,
        description=ONLINE_TABLE_DESCRIPTION,
        schema=ONLINE_TABLE_SCHEMA,
        table=ONLINE_TABLE,
        keys=ONLINE_TABLE_KEYS,
        partition_columns=ONLINE_TABLE_PARTITION,
        online=True,
    )

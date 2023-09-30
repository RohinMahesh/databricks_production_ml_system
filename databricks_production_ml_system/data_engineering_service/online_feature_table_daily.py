from databricks_production_ml_system.utils.constants import (
    ONLINE_TABLE,
    ONLINE_TABLE_DESCRIPTION,
    ONLINE_TABLE_KEYS,
    ONLINE_TABLE_PARTITION,
    ONLINE_TABLE_QUERY,
    ONLINE_TABLE_SCHEMA,
)
from databricks_production_ml_system.utils.helperfunctions import update_table
from pyspark.sql.functions import func


def feature_store_online_serving_update():
    """
    Executes daily update of the online feature table for serving
    """
    # Query Data
    data = spark.sql(ONLINE_TABLE_QUERY)
    online_features = data.groupby("CustomerNumber").agg(
        func.min("Feature1").alias("Feature1"),
        func.max("Feature2").alias("Feature2"),
        func.sum("Feature3").alias("Feature3"),
        func.last("Feature4").alias("Feature4"),
        func.last("CustomerState").alias("CustomerState"),
    )

    # Update Feature Table
    update_table(
        data=online_features,
        description=ONLINE_TABLE_DESCRIPTION,
        schema=ONLINE_TABLE_SCHEMA,
        table=ONLINE_TABLE,
        keys=ONLINE_TABLE_KEYS,
        partition_columns=ONLINE_TABLE_PARTITION,
        online=True,
    )

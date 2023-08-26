from databricks import feature_store
from databricks.feature_store.online_store_spec import AzureSqlServerSpec
import os
from pyspark.sql.functions import func
from pyspark.sql.types import IntegerType, DoubleType, StringType

from utils.constants import (
    ONLINE_STORE,
    ONLINE_TABLE_DESCRIPTION,
    ONLINE_TABLE,
    ONLINE_TABLE_KEYS,
    ONLINE_TABLE_PARTITION,
    ONLINE_TABLE_SCHEMA,
    ONLINE_TABLE_QUERY,
)
from utils.helperfunctions import publish_table, update_table

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

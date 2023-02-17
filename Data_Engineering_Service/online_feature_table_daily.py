from databricks import feature_store
from databricks.feature_store.online_store_spec import AzureSqlServerSpec
import os
from pyspark.sql.functions import func
from pyspark.sql.types import IntegerType, DoubleType, StringType

# Confgure Online Feature Table
fs = feature_store.FeatureStoreClient()
online_store = AzureSqlServerSpec(
    hostname="xxxx.database.windows.net",
    port=1433,
    database_name="",
    table_name="online_feature_table",
    read_secret_prefix="kv/online-feature-table",
    write_secret_prefix="kv/online-feature-table",
)

# Query Data
query = "SELECT * FROM offline.serving"
data = spark.sql(query)
online_features = data.groupby("CustomerNumber").agg(
    func.min("Feature1").alias("Feature1"),
    func.max("Feature2").alias("Feature2"),
    func.sum("Feature3").alias("Feature3"),
    func.last("Feature4").alias("Feature4"),
    func.last("CustomerState").alias("CustomerState"),
)

# Update Feature Table
if spark._jsparkSession().catalog().tableExists("online", "serving"):
    fs.write_table(name="online.serving", df=online_features, mode="overwrite")
else:
    fs.create_feature_table(
        name="online.serving",
        keys=["CustomerNumber"],
        features_df=online_features,
        partition_by=["CustomerState"]
        description="MRDS Online Feature Table",
    )

# Publish table to the online feature store
fs.publish_table(name="online.training", online_store=online_store, mode="overwrite")

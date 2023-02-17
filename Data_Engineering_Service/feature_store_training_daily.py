from databricks import feature_store
from datetime import datetime, timedelta
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pathlib import Path
import os
import random

# Load in table
path = ""
data = (
    spark.read.options(delimiter="\u0001", header=True)
    .csv(path)
    .distinct()
    .withColumn(
        "row_number",
        row_number().over(
            Window.partitionBy("ID", "Date", "CustomerNumber").orderBy(
                column("DateUpdated").desc()
            )
        ),
    )
    .where(column("row_number") == 1)
    .drop("row_number")
)

# Update Feature Table; if table does not exist, create one
fs = feature_store.FeatureStoreClient()
if spark._jsparkSession.catalog().tableExists("offline", "training"):
    fs.write_table(name="offline.training", df=data, mode="overwrite")
else:
    fs.create_feature_table(
        name="offline.training",
        keys=["ID", "Date", "CustomerNumber"],
        features_df=data,
        partition_columns=["CustomerState"],
        description="ARDS Feature Table",
    )
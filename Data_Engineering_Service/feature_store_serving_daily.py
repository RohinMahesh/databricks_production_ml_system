from databricks import feature_store
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pathlib import Path
import os
import random

# Get latest CSV file
path = ""
data = spark.read.options(delimiter="\u0001", header=True).csv(path)

# Update Feature Table; if table does not exist, create one
fs = feature_store.FeatureStoreClient()
if spark._jsparkSession.catalog().tableExists("offline", "serving"):
    fs.write_table(name="offline.serving", df=data, mode="overwrite")
else:
    fs.create_feature_table(
        name="offline.serving",
        keys=["ID", "Date", "CustomerNumber"],
        features_df=data,
        partition_columns=["CustomerState"],
        description="Model Serving Feature Table",
    )

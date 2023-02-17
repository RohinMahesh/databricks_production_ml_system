from databricks import feature_store
from datetime import datetime, timedelta
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pathlib import Path
import os
import random

# Get filtering condition
today = datetime.now()
cutoff = today - timedelta(days=2)
cutoff = cutoff.strftime("%Y-%m-%d")

# Load in table
path = ""
data = (
    spark.read.options(delimiter="\u0001", header=True)
    .csv(path)
    .withColumn(
        "row_number",
        row_number().over(
            Window.partitionBy("ID", "Date", "CustomerNumber").orderBy(
                column("Date").desc()
            )
        ),
    )
    .where(column("row_number") == 1)
    .drop("row_number")
)
data = data.filter(data.Date >= cutoff)

# Insert new records if new labeled data is available
if len(data.head(1)) > 0:
    # Select columns for downstream consumption
    data = data.select(
        "ID",
        "Date",
        "CustomerNumber",
        "CustomerState",
        "Feature1",
        "Feature2",
        "Feature3",
        "Feature4",
        "Target",
    )
    # Impute odd level of Feature4
    data = data.withColumn(
        "Feature4", when(data.Feature4 == "Unknown", "Status1").otherwise(data.Feature4)
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
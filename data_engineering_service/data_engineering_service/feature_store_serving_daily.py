from databricks import feature_store
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pathlib import Path
import os
import random

from utils.constants import (
    FILEPATH,
    DELIMITER,
    OFFLINE_TABLE_DESCRIPTION_SERVING,
    OFFLINE_TABLE_KEYS,
    OFFLINE_TABLE_PARTITION,
    OFFLINE_TABLE_SCHEMA,
    OFFLINE_TABLE_SERVING,
    OFFINE_TABLE_SERVING_DESCRIPTION,
)
from utils.helperfunctions import update_table

# Get latest CSV file
data = spark.read.options(delimiter=DELIMITER, header=True).csv(FILEPATH)

# Update feature table
update_table(
    data=data,
    description=OFFLINE_TABLE_DESCRIPTION_SERVING,
    schema=OFFLINE_TABLE_SCHEMA,
    table=OFFLINE_TABLE_SERVING,
    keys=OFFLINE_TABLE_KEYS,
    partition_columns=OFFLINE_TABLE_PARTITION,
)
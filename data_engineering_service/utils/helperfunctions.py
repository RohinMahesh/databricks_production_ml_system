from typing import List

from databricks import feature_store
from databricks.feature_store.online_store_spec import AzureSqlServerSpec
from pyspark.sql import DataFrame as SparkDataFrame

from constants import ONLINE_TABLE_SCHEMA, ONLINE_TABLE, ONLINE_STORE


def publish_table(
    schema: str = ONLINE_TABLE_SCHEMA,
    table: str = ONLINE_TABLE,
    online_store: AzureSqlServerSpec = ONLINE_STORE,
    mode: str = "overwrite",
):
    """
    Publishes online feature table

    :param schema: optional table schema name,
        defaults to ONLINE_TABLE_SCHEMA
    :param table: optional table name,
        defaults to ONLINE_TABLE
    :param online_store: optional database information for publishing,
        defaults to ONLINE_STORE
    :param mode: optional update method,
        defaults to "overwrite"
    :returns None
    """
    fs = feature_store.FeatureStoreClient()
    fs.publish_table(name=f"{schema}.{table}", online_store=online_store, mode=mode)


def update_table(
    data: SparkDataFrame,
    description: str,
    schema: str,
    table: str,
    keys: List[str],
    partition_columns: List[str],
    mode: str = "overwrite",
    online: bool = False,
):
    """
    Updates feature table

    :param data: data to update table with
    :param description: description of the table
    :param schema: table schema name
    :param table: table name
    :param keys: table key(s)
    :param partition_columns: column(s) for partitioning
    :param mode: optional update method,
        defaults to "overwrite"
    :param online: optional parameter for publishing table,
        defaults to False
    :returns None
    """
    fs = feature_store.FeatureStoreClient()
    if spark._jsparkSession.catalog().tableExists(schema, table):
        fs.write_table(name=f"{schema}.{table}", df=data, mode=mode)
    else:
        fs.create_feature_table(
            name=f"{schema}.{table}",
            keys=keys,
            features_df=data,
            partition_columns=partition_columns,
            description=description,
        )
    if online:
        publish_table()

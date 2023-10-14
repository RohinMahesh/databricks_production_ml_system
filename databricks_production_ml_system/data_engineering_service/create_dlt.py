from dataclasses import dataclass
from typing import List, Optional

import dlt
import pyspark.sql.functions as func
from databricks_production_ml_system.utils.constants import RESCUED_DATA_COLUMN


@dataclass
class CreateDLT:
    """
    Creates Delta Live Table (DLT)

    :param table_name: name of the upstream table
    :param date_col: date column name
    :param bronze_data_path: path to upstream data
    :param bronze_comment: comment for creation of bronze table
    :param bronze_customer_number: customer number column name
    :param silver_table: name of silver table
    :param silver_comment: comment for creation of silver table
    :param cloudfileformat: upstream file format for CDC
    :param keys: primary keys for upserting/deleting
    :param metadatacols: metadata columns to be dropped
    """

    table_name: str
    date_col: str
    bronze_data_path: str
    bronze_comment: str
    bronze_customer_number: str
    silver_table: str
    silver_comment: str
    cloudfileformat: str = "parquet"
    keys: Optional[List[str]] = None
    metadatacols: Optional[List[str]] = None

    def __post_init__(self):
        if not self.keys:
            object.__setattr__(self, "keys", [self.bronze_customer_number])
        if not self.metadatacols:
            object.__setattr__(self, "metadatacols", [RESCUED_DATA_COLUMN])

    def __call__(self):
        """
        Orchestrates DLT creation
        """
        self.create_bronze_table()
        self.expectations_and_transformations()
        self.run_cdc()
        self.create_silver_table()

    def create_bronze_table(self):
        """
        Creates Bronze table
        """

        @dlt.table(
            name=f"{self.table_name}_bronze",
            comment=self.bronze_comment,
            table_properties={"quality": "bronze"},
        )
        def upstream_data():
            return (
                spark.readStream.format("cloudFiles")
                .option("cloudFiles.format", self.cloudfileformat)
                .option("cloudFiles.inferColumnTypes", "true")
                .load(f"{self.bronze_data_path}/{self.table_name}")
            )

    def expectations_and_transformations(self):
        """
        Enforces data quality constraints and transformations
        """

        @dlt.view(
            name=f"{self.table_name}_pre_transformation",
            comment=f"Bronze table view {self.table_name}",
        )
        @dlt.expect_or_drop(
            f"{self.bronze_customer_number}",
            f"{self.bronze_customer_number} IS NOT NULL",
        )
        @dlt.expect_or_drop(RESCUED_DATA_COLUMN, f"{RESCUED_DATA_COLUMN} IS NULL")
        def bronze_transformation():
            return dlt.read_stream(f"{self.table_name}_bronze")

    def run_cdc(self):
        """
        Runs CDC and upserts incremental changes

        Apply changes parameters
            1. target: the table being materialized
            2. source: incoming CDC
            3. keys: primary keys for upserting/deleting
            4. sequence_by: duplication flag to get most recent value
            5. apply_as_deletes: DELETE condition
            6. except_column_list: metadata columns to be dropped
        """
        dlt.create_target_table(
            name=f"{self.table_name}_cdc",
            comment="CDC table",
            table_properties={"quality": "cdc"},
        )

        dlt.apply_changes(
            target=f"{self.table_name}_silver",
            source=f"{self.table_name}_pre_transformation",
            keys=self.keys,
            sequence_by=self.date_col,
            except_column_list=self.metadatacols,
        )

    def create_silver_table(self):
        """
        Creates Silver table for downstream consumption
        """

        @dlt.table(
            name=f"{self.table_name}_silver",
            comment=self.silver_comment,
            table_properties={"quality": "silver"},
        )
        def downstream_table():
            return dlt.read(f"{self.table_name}_silver")

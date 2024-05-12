from dataclasses import dataclass
from typing import List, Optional

import dlt

from databricks_production_ml_system.utils.constants import (
    BRONZE_COMMENT,
    CUSTOMER_COL,
    DATA_FORMAT,
    DATE_COL,
    DLT_TABLE_NAME,
    RESCUED_DATA_COLUMN,
    SILVER_COMMENT,
)
from databricks_production_ml_system.utils.file_paths import RAW_FILE_PATH


@dataclass
class CreateDLT:
    """
    Creates Delta Live Table (DLT)

    :param table_name: name of the upstream table
    :param date_col: date column name
    :param bronze_data_path: path to upstream data
    :param bronze_comment: comment for creation of bronze table
    :param customer_col: customer number column name
    :param silver_comment: comment for creation of silver table
    :param cloudfileformat: upstream file format for CDC
    :param keys: primary keys for upserting/deleting
    :param metadatacols: metadata columns to be dropped
    """

    table_name: str = DLT_TABLE_NAME
    date_col: str = DATE_COL
    bronze_data_path: str = RAW_FILE_PATH
    bronze_comment: str = BRONZE_COMMENT
    customer_col: str = CUSTOMER_COL
    silver_comment: str = SILVER_COMMENT
    cloudfileformat: str = DATA_FORMAT
    keys: Optional[List[str]] = None
    metadatacols: Optional[List[str]] = None

    def __post_init__(self):
        if not self.keys:
            object.__setattr__(self, "keys", [self.customer_col])
        if not self.metadatacols:
            object.__setattr__(self, "metadatacols", [RESCUED_DATA_COLUMN])

    def __call__(self):
        """
        Orchestrates DLT creation
        """
        self._create_bronze_table()
        self._expectations_and_transformations()
        self._run_cdc()
        self._create_silver_table()

    def _create_bronze_table(self) -> None:
        """
        Creates Bronze table

        :returns None
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

    def _expectations_and_transformations(self) -> None:
        """
        Enforces data quality constraints and transformations

        :returns None
        """

        @dlt.view(
            name=f"{self.table_name}_pre_transformation",
            comment=f"Bronze table view {self.table_name}",
        )
        @dlt.expect_or_drop(
            f"{self.customer_col}",
            f"{self.customer_col} IS NOT NULL",
        )
        @dlt.expect_or_drop(RESCUED_DATA_COLUMN, f"{RESCUED_DATA_COLUMN} IS NULL")
        def bronze_transformation():
            return dlt.read_stream(f"{self.table_name}_bronze")

    def _run_cdc(self) -> None:
        """
        Runs CDC and upserts incremental changes

        Apply changes parameters
            1. target: the table being materialized
            2. source: incoming CDC
            3. keys: primary keys for upserting/deleting
            4. sequence_by: duplication flag to get most recent value
            5. apply_as_deletes: DELETE condition
            6. except_column_list: metadata columns to be dropped

        :returns None
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

    def _create_silver_table(self) -> None:
        """
        Creates Silver table for downstream consumption

        :returns None
        """

        @dlt.table(
            name=f"{self.table_name}_silver",
            comment=self.silver_comment,
            table_properties={"quality": "silver"},
        )
        def downstream_table():
            return dlt.read(f"{self.table_name}_silver")

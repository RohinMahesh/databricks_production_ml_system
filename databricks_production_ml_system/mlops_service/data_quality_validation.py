import uuid
from datetime import datetime, timedelta

import pandas as pd
import pyspark.sql.functions as func
from databricks import feature_store
from databricks_production_ml_system.utils.constants import (
    CHECKPOINT_NAME,
    DATA_SOURCE_CONFIG,
    EXPECTATION_SUITE_NAME,
    NUMERICAL_COLS,
)
from databricks_production_ml_system.utils.file_paths import GE_ROOT_DIRECTORY
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)
from ruamel import yaml


def run_validation():
    """
    Runs Great Expectations data quality validation
    """
    # Load data for validation
    current_date = datetime.today()
    cutoff = current_date - timedelta(days=1)
    cutoff = cutoff.strftime("%Y-%m-%d")

    fs = feature_store.FeatureStoreClient()
    query = "SELECT * FROM example.training WHERE Date >= {0}".format(cutoff)
    data = spark.sql(query)
    data = data.filter(data.Date >= cutoff)

    # Set up configuration
    data_context_config = DataContextConfig(
        store_backend_defaults=FilesystemStoreBackendDefaults(
            root_directory=GE_ROOT_DIRECTORY
        )
    )
    base_data_context = BaseDataContext(project_config=data_context_config)

    base_data_context.test_yaml_config(yaml.dump(DATA_SOURCE_CONFIG))
    base_data_context.add_datasource(**DATA_SOURCE_CONFIG)

    unique_id = uuid.uuid4()
    curr_date = datetime.date.today.strftime("%Y-%m-%d")
    unique_runid = f"{unique_id}_{curr_date}"

    batch_request = RuntimeBatchRequest(
        datasource_name="example",
        data_connector_name="example_connector",
        data_asset_name="example_data",
        batch_identifiers={"run_id": unique_runid},
        runtime_parameters={"data": data},
    )

    # Define and save expectations
    base_data_context.create_expectation_suite(
        expectation_suite_name=EXPECTATION_SUITE_NAME, overwrite_existing=True
    )
    validator = base_data_context.get_validator(
        batch_request=batch_request, expectation_suite_name=EXPECTATION_SUITE_NAME
    )

    validator.expect_column_values_to_be_in_set(
        column="feature4", value_set=NUMERICAL_COLS
    )

    validator.expect_column_values_to_not_be_null(column="customer_number")
    validator.expect_column_values_to_not_be_null(column="feature1")
    validator.expect_column_values_to_not_be_null(column="feature2")
    validator.expect_column_values_to_not_be_null(column="feature3")
    validator.expect_column_values_to_not_be_null(column="feature4")
    validator.expect_column_values_to_not_be_null(column="target")

    validator.save_expectation_suite(discard_failed_expectations=False)

    # Configure and run checkpoint
    checkpoint_config = {
        "name": CHECKPOINT_NAME,
        "config_version": 0.1,
        "class_name": "SimpleCheckpoint",
        "run_name_template": f"template_{current_date}",
    }
    checkpoint = base_data_context.test_yaml_config(yaml.dump(checkpoint_config))
    base_data_context.add_checkpoint(**checkpoint_config)
    checkpoint_result = base_data_context.run_checkpoint(
        checkpoint_name=CHECKPOINT_NAME,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": EXPECTATION_SUITE_NAME,
            }
        ],
    )

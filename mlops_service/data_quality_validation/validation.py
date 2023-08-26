import uuid
from datetime import datetime, timedelta

import pandas as pd
import pyspark.sql.functions as func
from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)
from ruamel import yaml

# Load data for validation
current_date = datetime.today()
cutoff = current_date - timedelta(days=1)
cutoff = cutoff.strftime("%Y-%m-%d")

fs = feature_store.FeatureStoreClient()
query = "SELECT * FROM example.training WHERE Date >= {0}".format(cutoff)
data = spark.sql(query)
data = data.filter(data.Date >= cutoff)

# Set up configuration
root_directory = "/dbfs/great_expectations/"
data_context_config = DataContextConfig(
    store_backend_defaults=FilesystemStoreBackendDefaults(root_directory=root_directory)
)
base_data_context = BaseDataContext(project_config=data_context_config)

data_source_config = {
    "name": "example",
    "class_name": "Datasource",
    "execution_engine": {"class_name": "SparkDFExecutionEngine"},
    "data_connectors": {
        "example_connector": {
            "module_name": "great_expectations.datasource.data_connector",
            "class_name": "RuntimeDataConnector",
            "batch_identifiers": [
                "run_id",
            ],
        }
    },
}

base_data_context.test_yaml_config(yaml.dump(data_source_config))
base_data_context.add_datasource(**data_source_config)

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
expectation_suite_name = "example_expectations"
base_data_context.create_expectation_suite(
    expectation_suite_name=expectation_suite_name, overwrite_existing=True
)
validator = base_data_context.get_validator(
    batch_request=batch_request, expectation_suite_name=expectation_suite_name
)

feature_four_levels = ["Status1", "Status2", "Status3"]
validator.expect_column_values_to_be_in_set(
    column="Feature4", value_set=feature_four_levels
)

validator.expect_column_values_to_not_be_null(column="CustomerNumber")
validator.expect_column_values_to_not_be_null(column="Feature1")
validator.expect_column_values_to_not_be_null(column="Feature2")
validator.expect_column_values_to_not_be_null(column="Feature3")
validator.expect_column_values_to_not_be_null(column="Feature4")
validator.expect_column_values_to_not_be_null(column="Target")

validator.save_expectation_suite(discard_failed_expectations=False)

# Configure and run checkpoint
checkpoint_name = "example_checkpoint_name"
checkpoint_config = {
    "name": checkpoint_name,
    "config_version": 0.1,
    "class_name": "SimpleCheckpoint",
    "run_name_template": f"template_{current_date}",
}
checkpoint = base_data_context.test_yaml_config(yaml.dump(checkpoint_config))
base_data_context.add_checkpoint(**checkpoint_config)
checkpoint_result = base_data_context.run_checkpoint(
    checkpoint_name=checkpoint_name,
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": expectation_suite_name,
        }
    ],
)

import ast
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import mlflow
import mlflow.sklearn
import pandas as pd
import pyspark.sql.functions as func
from databricks import feature_store
from databricks.feature_store.online_store_spec import AzureSqlServerSpec
from databricks_production_ml_system.utils.constants import (
    CATEGORICAL_COLS,
    DATE_COL,
    EXPERIMENT_NAME,
    HYPERPARAMS,
    MODEL_NAME,
    NUMERICAL_COLS,
    ONLINE_STORE,
    ONLINE_TABLE,
    ONLINE_TABLE_SCHEMA,
    RUN_NAME,
    TARGET,
    USER,
)
from databricks_production_ml_system.utils.file_paths import HTML_DIRECTORY
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame as SparkDataFrame


def register_mlflow(
    data: pd.DataFrame,
    model: Pipeline,
    experiment_name: str = EXPERIMENT_NAME,
    run_name: str = RUN_NAME,
    model_name: str = MODEL_NAME,
    user: str = USER,
    parameters: dict = HYPERPARAMS,
    stage: str = "Staging",
):
    """
    Register ML model artifacts in MLflow

    :param data: reference data for signature
    :param model: ML pipeline
    :param experiment_name: optional name of the experiment,
        defaults to EXPERIMENT_NAME
    :param run_name: optional name of the ML run,
        defaults to RUN_NAME
    :param model_name: optional name of your the artifact,
        defaults to MODEL_NAME
    :param user: optional name of the ML engineer registering the ML artifact,
        defaults to USER
    :param parameters: optional hyperparameters for the ML pipeline,
        defaults to HYPERPARAMETERS
    :param stage: optional MLflow environment the ML artifact has been promoted to,
        defaults to "Staging"
    :returns None
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        # Define signature
        signature = infer_signature(data, model.predict_proba(data))

        # Log model
        mlflow.sklearn.log_model(model, model_name, signature=signature)

        # Log parameters
        mlflow.log_params(parameters)

        # Tag the run_id
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("run_id", run_id)

    mlflow.end_run()

    # Get experiment information
    experiment_info = mlflow.get_experiment_by_name(name=experiment_name)
    experiment_id = experiment_info.experiment_id

    # Get experiment metadata
    exp_metadata = mlflow.search_runs([experiment_id])
    exp_metadata = exp_metadata.sort_values(by=["end_time"], ascending=False)
    run_id = exp_metadata.loc[0, "run_id"]
    model_uri = exp_metadata.loc[0, "artifact_uri"]

    # Model registry
    details = mlflow.register_model(
        model_uri=model_uri + f"/{model_name}", name=model_name
    )
    model_version = details.version

    # Promote registered model to staging
    client = MlflowClient()
    current_date = datetime.today().date().strftime("%Y-%m-%d")
    description = f"The model version {model_version} as transitioned to {stage} on {current_date} by {user}"

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage,
        archive_existing_versions=True,
    )

    client.update_model_version(
        name=model_name, version=model_version, description=description
    )


def load_mlflow(model_name: str, stage: str):
    """
    Gets production model from MLflow

    :param model_name: name of your model
    :param stage: MLflow stage
    :returns model: model artifact
    """
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    return model


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


def create_drift_report(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    numerical_f: List[str] = NUMERICAL_COLS,
    categorical_f: List[str] = CATEGORICAL_COLS,
    num_stattest: str = "ks",
    cat_stattest: str = "psi",
    num_stattest_threshold: float = 0.15,
    cat_stattest_threshold: float = 0.15,
):
    """
    Create drift report

    :param reference: reference data for drift evaluation
    :param comparison: comparison data for drift evaluation
    :param numerical_f: optional list of numerical covariates,
        defaults to NUMERICAL_COLS
    :param categorical_f: list of categorical covariates,
        defaults to CATEGORICAL_COLS
    :param num_stattest: optional statistical test for numerical covariates,
        defaults to 'ks'
    :param cat_stattest: optional statistical test for categorical covariates,
        defaults to 'psi'
    :param num_stattest_threshold: optional threshold for numerical tests,
        defaults to 0.15
    :param cat_stattest_threshold: optional threshold for categorical tests,
        defaults to 0.15
    :returns drift_report: dictionary containing drift report
    """
    # Define dictionary of statistical test per covariate
    per_column_stattest = {
        **{feature: num_stattest for feature in numerical_f},
        **{feature: cat_stattest for feature in categorical_f},
    }

    # Define dictionary of thresholds per covariate
    per_column_stattest_threshold = {
        **{feature: num_stattest_threshold for feature in numerical_f},
        **{feature: cat_stattest_threshold for feature in categorical_f},
    }

    # Define testing suite
    data_drift = TestSuite(
        tests=[
            DataDriftTestPreset(
                per_column_stattest=per_column_stattest,
                per_column_stattest_threshold=per_column_stattest_threshold,
            ),
        ]
    )

    # Initialize drift report
    drift_report = {feature: None for feature in reference.columns}

    # Calculate drift metrics and convert to dictionary
    data_drift.run(reference_data=comparison, current_data=reference)
    data_drift = data_drift.as_dict()

    # Populate drift report with status of test
    for feature in reference.columns:
        drift_report[feature] = data_drift["tests"][0]["parameters"]["features"][
            feature
        ]["detected"]

    # Engineer any_drift for downstream orchestration
    drift_report["any_drift"] = (
        False if data_drift["summary"]["all_passed"] == True else True
    )
    return drift_report


def get_drift_data(beginning, mid):
    """
    Generates reference and comparison data using 14 day window

    :param beginning: beginning time window to filter/extract data
    :param mid: middle time window to filter/extract data
    :returns comparison_data: data used for comparison
    :returns reference_data: data used for reference
    """

    # Get data for comparison
    today = datetime.now()
    cutoff = today - timedelta(days=60)
    cutoff = cutoff.strftime("%Y-%m-%d")
    fs = feature_store.FeatureStoreClient()
    query = "SELECT * FROM example.training WHERE Date >= {0}".format(cutoff)
    data = spark.sql(query)

    # Identify reference and comparison data
    comparison_data = data.filter(
        (data.Date >= beginning.strftime("%Y-%m-%d"))
        & (data.Date < mid.strftime("%Y-%m-%d"))
    ).toPandas()
    reference_data = data.filter(data.Date >= mid.strftime("%Y-%m-%d")).toPandas()

    return comparison_data, reference_data

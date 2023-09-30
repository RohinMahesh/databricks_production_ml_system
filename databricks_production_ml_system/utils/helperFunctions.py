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
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.options import DataDriftOptions
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame as SparkDataFrame
from utils.constants import (
    CATEGORICAL_COLS,
    CONTAINER,
    D_TIME,
    EXPERIMENT_NAME,
    HTML_DIRECTORY,
    HYPERPARAMS,
    MODEL_NAME,
    MOUNT_NAME,
    NUMERICAL_COLS,
    ONLINE_STORE,
    ONLINE_TABLE,
    ONLINE_TABLE_SCHEMA,
    RUN_NAME,
    STORAGE,
    STORAGE_ACC_KEY,
    TARGET,
    USER,
)


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


def calculate_drift(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    target: str = TARGET,
    numerical_f: List[str] = NUMERICAL_COLS,
    categorical_f: List[str] = CATEGORICAL_COLS,
    statTest: str = "psi",
    thresh: float = 0.15,
    prediction: str = None,
    Id: str = None,
    d_time: str = D_TIME,
):
    """
    Calculates drift

    :param reference: reference data for drift evaluation
    :param comparison: comparison data for drift evaluation
    :param target: optional target variable,
        defaults to TARGET
    :param numerical_f: optional list of numerical covariates,
        defaults to NUMERICAL_COLS
    :param categorical_f: list of categorical covariates,
        defaults to CATEGORICAL_COLS
    :param statTest: optional statistical test,
        defaults to "psi"
    :param thresh: optional threshold for the statistical test,
        defaults to 0.15
    :param prediction: optional column with ML model predictions,
        defaults to None
    :param Id: optional unique identifier,
        defaults to None
    :param d_time: optional datetime column,
        defaults to D_TIME
    :returns drift_profile: dictionary containing results of statistical tests
    """
    # Configure mapping
    columnMapping = ColumnMapping()
    columnMapping.task = "classification"
    columnMapping.target = target
    columnMapping.prediction = prediction
    columnMapping.id = Id
    columnMapping.datetime = d_time
    columnMapping.numerical_features = numerical_f
    columnMapping.categorical_features = categorical_f

    # Create drift metric and threshold
    opt = DataDriftOptions(threshold=thresh, feature_stattest_func=statTest)
    drift_profile = Profile(sections=[DataDriftProfileSection()], options=[opt])

    # Calculate drift report
    drift_profile.calculate(reference, comparison, column_mapping=columnMapping)

    # Write index.html file to Blob container for static webpage
    drift_report_html = Report(
        metrics=[
            DatasetDriftMetric(threshold=0.15, options=opt),
            DataDriftTable(options=opt),
        ]
    )
    drift_report_html.run(
        reference_data=reference, current_data=comparison, column_mapping=columnMapping
    )

    # Store locally to tmp directory in DBFS
    drift_report_html.save_html(HTML_DIRECTORY)

    # Mount blob container and then move file into container
    dbutils.fs.mount(
        source=f"wasbs://{CONTAINER}@{STORAGE}.blob.core.windows.net",
        mount_point=f"/mnt/{MOUNT_NAME}",
        extra_configs={
            f"fs.azure.account.key.{STORAGE}.blob.core.windows.net": f"{STORAGE_ACC_KEY}"
        },
    )
    dbutils.fs.cp(
        "dbfs:/FileStore/tmp/index.html",
        f"dbfs:/mnt/{MOUNT_NAME}/index.html",
    )
    # Unmount container
    dbutils.fs.unmount(f"/mnt/{MOUNT_NAME}")
    return drift_profile


def create_drift_report(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    target: str = TARGET,
    numerical_f: List[str] = NUMERICAL_COLS,
    categorical_f: List[str] = CATEGORICAL_COLS,
    statTest: str = "psi",
    thresh: float = 0.15,
    prediction: str = None,
    Id: str = None,
    d_time: str = D_TIME,
):
    """
    Create drift report

    :param reference: reference data for drift evaluation
    :param comparison: comparison data for drift evaluation
    :param target: optional target variable,
        defaults to TARGET
    :param numerical_f: optional list of numerical covariates,
        defaults to NUMERICAL_COLS
    :param categorical_f: list of categorical covariates,
        defaults to CATEGORICAL_COLS
    :param statTest: optional statistical test,
        defaults to "psi"
    :param thresh: optional threshold for the statistical test,
        defaults to 0.15
    :param prediction: optional column with ML model predictions,
        defaults to None
    :param Id: optional unique identifier,
        defaults to None
    :param d_time: optional datetime column,
        defaults to D_TIME
    :returns drift_report: dictionary containing drift report
    """
    drift_profile = calculate_drift(
        reference,
        comparison,
    )
    # Convert to JSON
    drift_profile = drift_profile.json()

    # Remove single quotes
    drift_profile = drift_profile.replace("'", "")
    drift_profile = json.loads(drift_profile)

    # Select covariates for extraction
    covariates = list(comparison.columns)
    for_selection = numerical_f
    for_selection.extend(categorical_f)
    for_selection.append(target)
    covariates = [x for x in covariates for x in for_selection]
    drift_report = {x: list() for x in covariates}
    any_drift = False

    # Extract covariates from drift report
    for feature in list(drift_report.keys()):
        drift_report[feature] = {
            "drift_detected": drift_profile["data_drift"]["data"]["metrics"][feature][
                "drift_detected"
            ],
            "stattest_name": drift_profile["data_drift"]["data"]["metrics"][feature][
                "stattest_name"
            ],
            "drift_score": round(
                drift_profile["data_drift"]["data"]["metrics"][feature]["drift_score"],
                4,
            ),
        }
        if drift_report[feature]["drift_detected"]:
            any_drift = True

    # Assign any_drift
    drift_report["any_drift"] = any_drift
    return drift_report


def get_drift_data(beginning, mid):
    """Generates reference and comparison data using 14 day window

    Args:
        beginning (String):
            Filtering condition for extracting data
        mid (String):
            Filtering condition for extracting data

    Returns:
        comparison_data (Pandas DataFrame):
            Data used for comparison
        reference_data (Pandas DataFrame):
            Data used for reference
    """

    # Get data for comparison
    today = datetime.now()
    cutoff = today - timedelta(days=60)
    cutoff = cutoff.strftime("%Y-%m-%d")
    fs = feature_store.FeatureStoreClient()
    cutoff = "SELECT * FROM example.training WHERE Date >= {0}".format(cutoff)
    data = spark.sql(query)

    # Identify reference and comparison data
    comparison_data = data.filter(
        (data.Date >= beginning.strftime("%Y-%m-%d"))
        & (data.Date < mid.strftime("%Y-%m-%d"))
    ).toPandas()
    reference_data = data.filter(data.Date >= mid.strftime("%Y-%m-%d")).toPandas()

    return comparison_data, reference_data

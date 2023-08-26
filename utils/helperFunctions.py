from dataclasses import dataclass
from typing import Dict, List

from databricks import feature_store
from databricks.feature_store.online_store_spec import AzureSqlServerSpec
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import pandas as pd

import pyspark.sql.functions as func
from pyspark.sql import DataFrame as SparkDataFrame

from utils.constants import (
    EXPERIMENT_NAME,
    HYPERPARAMS,
    MODEL_NAME,
    ONLINE_TABLE_SCHEMA,
    ONLINE_TABLE,
    ONLINE_STORE,
    RUN_NAME,
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

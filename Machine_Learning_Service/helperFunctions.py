from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature


def register_mlflow(
    experiment_name,
    run_name,
    model_name,
    user,
    data,
    model,
    parameters,
    stage="Staging",
):
    """Register ML model artifacts in MLflow

    Args:
        experiment_name (String):
            Name of your experiment
        run_name (String):
            Name of your ML run
        model_name (String):
            Name of your ML artifact
        user (String):
            Name of ML engineer registering ML artifact
        data (Pandas DataFrame):
            Reference data for signature
        model (Scikit-learn Pipeline):
            ML pipeline
        parameters (Dict):
            Hyperparameters for ML pipeline
        stage (String):
            MLflow environment ML artifact has been promoted to
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


def load_mlflow(model_name, stage):
    """Get production model from MLflow

    Args:
        model_name (String):
            Name of your model
        stage (String):
            Stage in MLflow

    Returns:
        model: Production model artifact
    """
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    return model

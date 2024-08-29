import os
import tempfile
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from databricks_production_ml_system.utils.helpers import (
    check_data_exists,
    create_drift_report,
    get_drift_data,
    register_mlflow,
)
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from sklearn.pipeline import Pipeline

unittest_spark = (
    SparkSession.builder.appName("unit-test-helpers").master("local[*]").getOrCreate()
)


@pytest.mark.parametrize(
    "data, model, experiment_name, run_name, model_name, user, parameters, stage",
    [
        (
            pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]}),
            mock.Mock(spec=Pipeline),
            "test_experiment",
            "test_run",
            "test_model",
            "test_user",
            {"param1": 0.1, "param2": 0.2},
            "Staging",
        ),
    ],
)
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.set_experiment")
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.start_run")
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.active_run")
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.sklearn.log_model")
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.log_params")
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.set_tag")
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.end_run")
@mock.patch(
    "databricks_production_ml_system.utils.helpers.mlflow.get_experiment_by_name"
)
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.search_runs")
@mock.patch("databricks_production_ml_system.utils.helpers.mlflow.register_model")
@mock.patch("databricks_production_ml_system.utils.helpers.MlflowClient")
def test_register_mlflow(
    mock_MlflowClient,
    mock_register_model,
    mock_search_runs,
    mock_get_experiment_by_name,
    mock_end_run,
    mock_set_tag,
    mock_log_params,
    mock_log_model,
    mock_active_run,
    mock_start_run,
    mock_set_experiment,
    data,
    model,
    experiment_name,
    run_name,
    model_name,
    user,
    parameters,
    stage,
):
    model.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8]])

    mock_run = mock.Mock()
    mock_run.info.run_id = "test_run_id"
    mock_active_run.return_value = mock_run

    with tempfile.TemporaryDirectory() as tempdir:
        mock_get_experiment_by_name.return_value.experiment_id = "test_experiment_id"
        mock_search_runs.return_value = pd.DataFrame(
            {
                "run_id": ["test_run_id"],
                "artifact_uri": [tempdir],
                "end_time": [pd.Timestamp("2023-08-01")],
            }
        )
        mock_register_model.return_value.version = "1"

        register_mlflow(
            data, model, experiment_name, run_name, model_name, user, parameters, stage
        )

        mock_set_experiment.assert_called_once_with(experiment_name)
        mock_start_run.assert_called_once_with(run_name=run_name)
        mock_log_model.assert_called_once_with(model, model_name, signature=mock.ANY)
        mock_log_params.assert_called_once_with(parameters)
        mock_set_tag.assert_called_once_with("run_id", "test_run_id")
        mock_end_run.assert_called_once()
        mock_register_model.assert_called_once_with(
            model_uri=os.path.join(tempdir, model_name), name=model_name
        )
        mock_MlflowClient().transition_model_version_stage.assert_called_once_with(
            name=model_name, version="1", stage=stage, archive_existing_versions=True
        )
        mock_MlflowClient().update_model_version.assert_called_once_with(
            name=model_name, version="1", description=mock.ANY
        )


@pytest.mark.parametrize(
    "reference, comparison, expected_output",
    [
        (
            pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}),
            pd.DataFrame({"feature1": [1, 2, 3], "feature2": [7, 8, 9]}),
            {"feature1": False, "feature2": True, "any_drift": True},
        ),
        (
            pd.DataFrame({"feature1": [1, 1, 1], "feature2": [4, 4, 4]}),
            pd.DataFrame({"feature1": [1, 1, 1], "feature2": [4, 4, 4]}),
            {"feature1": False, "feature2": False, "any_drift": False},
        ),
    ],
)
def test_create_drift_report(reference, comparison, expected_output):
    output = create_drift_report(reference, comparison)
    assert output == expected_output


@pytest.mark.parametrize(
    "beginning, mid, mock_data, expected_comparison_data, expected_reference_data",
    [
        (
            datetime(2024, 1, 1),
            datetime(2024, 1, 15),
            pd.DataFrame(
                {
                    "Date": ["2024-01-01", "2024-01-10", "2024-01-20"],
                    "feature1": [1, 2, 3],
                }
            ),
            pd.DataFrame({"Date": ["2024-01-01", "2024-01-10"], "feature1": [1, 2]}),
            pd.DataFrame({"Date": ["2024-01-20"], "feature1": [3]}),
        ),
    ],
)
@mock.patch(
    "databricks_production_ml_system.utils.helpers.feature_store.FeatureStoreClient"
)
def test_get_drift_data(
    mock_fs_client,
    beginning,
    mid,
    mock_data,
    expected_comparison_data,
    expected_reference_data,
):
    schema = StructType(
        [
            StructField("Date", StringType(), True),
            StructField("feature1", IntegerType(), True),
        ]
    )

    mock_spark_df = unittest_spark.createDataFrame(
        mock_data.to_dict(orient="records"), schema=schema
    )

    with mock.patch.object(unittest_spark, "sql", return_value=mock_spark_df):
        comparison_data, reference_data = get_drift_data(
            beginning, mid, spark=unittest_spark
        )

        expected_comparison_data["Date"] = expected_comparison_data["Date"].astype(str)
        expected_comparison_data["feature1"] = expected_comparison_data[
            "feature1"
        ].astype("int64")
        expected_reference_data["Date"] = expected_reference_data["Date"].astype(str)
        expected_reference_data["feature1"] = expected_reference_data[
            "feature1"
        ].astype("int64")

        comparison_data_dict = comparison_data.to_dict(orient="records")
        expected_comparison_data_dict = expected_comparison_data.to_dict(
            orient="records"
        )
        reference_data_dict = reference_data.to_dict(orient="records")
        expected_reference_data_dict = expected_reference_data.to_dict(orient="records")

        assert (
            comparison_data_dict == expected_comparison_data_dict
        ), f"Comparison data mismatch: {comparison_data_dict} != {expected_comparison_data_dict}"
        assert (
            reference_data_dict == expected_reference_data_dict
        ), f"Reference data mismatch: {reference_data_dict} != {expected_reference_data_dict}"


@pytest.mark.parametrize(
    "f_path, expected",
    [
        ("/valid/path", True),
        ("/empty/path", False),
        ("/invalid/path", False),
    ],
)
def test_check_data_exists(f_path, expected):
    with tempfile.TemporaryDirectory() as temp_dir:
        if f_path == "/valid/path":
            f_path = temp_dir

            file_paths = []
            for i in range(3):
                file_path = os.path.join(temp_dir, f"file{i}.txt")
                with open(file_path, "w") as f:
                    f.write("Some content")
                file_paths.append(file_path)
        elif f_path == "/empty/path":
            f_path = temp_dir
            file_paths = []
        else:
            file_paths = None

        mock_dbutils = mock.Mock()

        if file_paths is not None:
            mock_dbutils.fs.ls.return_value = [f"mock://{fp}" for fp in file_paths]
        else:
            mock_dbutils.fs.ls.side_effect = Exception("Path not found")

        result = check_data_exists(f_path, dbutils=mock_dbutils)

        assert result == expected

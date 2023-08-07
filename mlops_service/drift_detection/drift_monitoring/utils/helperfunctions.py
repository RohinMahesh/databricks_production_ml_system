from typing import List

from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from datetime import datetime, timedelta
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.options import DataDriftOptions
from evidently.report import Report
import json
import pandas as pd

from utils.constants import (
    CATEGORICAL_COLS,
    CONTAINER,
    D_TIME,
    HTML_DIRECTORY,
    MOUNT_NAME,
    NUMERICAL_COLS,
    STORAGE,
    STORAGE_ACC_KEY,
    TARGET,
)


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

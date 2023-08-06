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


def calculate_drift(
    reference,
    comparison,
    target,
    numerical_f,
    categorical_f,
    statTest="psi",
    thresh=0.15,
    prediction=None,
    Id=None,
    d_time=None,
):
    """Calculates Drift

    Args:
        reference (Pandas DataFrame):
            Reference data for drift evaluation
        comparison (Pandas DataFrame):
            Comparison data for drift evaluation
        target (String):
            Target variable
        numerical_f (List):
            List of numerical covariates
        categorical_f (List):
            List of categorical covariates
        statTest (String, optional):
            Statistical test. Defaults to "psi".
        thresh (Float, optional):
            Threshold for statistical test. Defaults to 0.15.
        prediction (String, optional):
            Column with ML model predictions. Defaults to None.
        Id (String, optional):
            Unique identifier. Defaults to None.
        d_time (String, optional):
            Datetime column. Defaults to None.

    Returns:
        drift_profile

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
    drift_report_html.save_html("dbfs:/FileStore/tmp/index.html")

    # Mount blob container and then move file into container
    storage = ""
    container = ""
    mount_name = ""
    storage_acc_key = ""

    dbutils.fs.mount(
        source=f"wasbs://{container}@{storage}.blob.core.windows.net",
        mount_point=f"/mnt/{mount_name}",
        extra_configs={
            f"fs.azure.account.key.{storage}.blob.core.windows.net": f"{storage_acc_key}"
        },
    )
    dbutils.fs.cp(
        "dbfs:/FileStore/tmp/index.html",
        f"dbfs:/mnt/{mount_name}/index.html",
    )
    # Unmount container
    dbutils.fs.unmount(f"/mnt/{mount_name}")
    return drift_profile


def create_drift_report(
    reference,
    comparison,
    target,
    numerical_f,
    categorical_f,
    statTest="psi",
    thresh=0.15,
    prediction=None,
    Id=None,
    d_time=None,
):
    """Create Drift Report

    Args:
        reference (Pandas DataFrame):
            Reference data for drift evaluation
        comparison (Pandas DataFrame):
            Comparison data for drift evaluation
        target (String):
            Target variable
        numerical_f (List):
            List of numerical covariates
        categorical_f (List):
            List of categorical covariates
        statTest (String, optional):
            Statistical test. Defaults to "psi".
        thresh (Float, optional):
            Threshold for statistical test. Defaults to 0.15.
        prediction (String, optional):
            Column with ML model predictions. Defaults to None.
        Id (String, optional):
            Unique identifier. Defaults to None.
        d_time (String, optional):
            Datetime column. Defaults to None.

    Returns:
        drift_report (Dict):
            JSON structure of drift report
    """
    drift_profile = calculate_drift(
        reference,
        comparison,
        target,
        numerical_f,
        categorical_f,
        statTest,
        thresh,
        prediction,
        Id,
        d_time,
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

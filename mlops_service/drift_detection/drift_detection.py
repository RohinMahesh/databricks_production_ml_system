import json
from datetime import datetime, timedelta

import pandas as pd
from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.options import DataDriftOptions
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from utils.constants import (
    BEGINNING,
    CATEGORICAL_COLS,
    CONTAINER,
    D_TIME,
    HTML_DIRECTORY,
    MID,
    MOUNT_NAME,
    NUMERICAL_COLS,
    STORAGE,
    STORAGE_ACC_KEY,
    TARGET,
)
from utils.helperfunctions import calculate_drift, create_drift_report, get_drift_data


def drift_detection():
    """
    Gets drift report for downstream consumption

    :returns drift_report: dictionary containing drift status and values
    """
    # Extract comparison and reference data
    comparison_data, reference_data = get_drift_data(BEGINNING, MID)

    # Get drift report
    drift_report = create_drift_report(
        reference=reference_data,
        comparison=comparison_data,
    )
    return drift_report

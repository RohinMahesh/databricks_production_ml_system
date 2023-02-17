from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from datetime import datetime, timedelta
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.options import DataDriftOptions
from evidently.report import Report
from helperFunctions import calculate_drift, create_drift_report, get_drift_data
import json
import pandas as pd

# Identify filtering conditions
today = datetime.now()
beginning = today - timedelta(days=14)
mid = today - timedelta(days=7)

# Extract comparison and reference data
comparison_data, reference_data = get_drift_data(beginning, mid)

# Define required input for drift detection
target = "Target"
numerical = ["Feature1", "Feature2", "Feature3"]
categorical = ["Feature4"]
d_time = "Date"

# Get drift report
drift_report = create_drift_report(
    reference=reference_data,
    comparison=comparison_data,
    target=target,
    numerical_f=numerical,
    categorical_f=categorical,
    d_time=d_time,
)
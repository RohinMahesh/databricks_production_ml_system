import pandas as pd
from databricks_production_ml_system.utils.constants import BEGINNING, MID
from databricks_production_ml_system.utils.helperfunctions import (
    create_drift_report,
    get_drift_data,
)


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

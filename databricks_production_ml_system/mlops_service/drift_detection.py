from typing import Any, Dict

from databricks_production_ml_system.machine_learning_service.training_pipeline import (
    TrainingPipeline,
)
from databricks_production_ml_system.utils.constants import BEGINNING, MID
from databricks_production_ml_system.utils.helpers import (
    create_drift_report,
    get_drift_data,
)


def drift_detection() -> Dict[str, Any]:
    """
    Creates drift report and evaluates status for model retraining

    :returns drift_report: dictionary containing drift status and values
    """
    # Extract comparison and reference data
    comparison_data, reference_data = get_drift_data(BEGINNING, MID)

    # Get drift report
    drift_report = create_drift_report(
        reference=reference_data,
        comparison=comparison_data,
    )

    # If there is any drift, trigger model retraining
    if drift_report["any_drift"] == True:
        TrainingPipeline().train_and_register_model()

    return drift_report

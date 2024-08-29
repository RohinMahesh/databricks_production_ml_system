from unittest.mock import MagicMock, patch

import pytest
from databricks_production_ml_system.mlops_service.drift_detection import (
    drift_detection,
)
from databricks_production_ml_system.utils.constants import BEGINNING, MID


@pytest.mark.parametrize(
    "drift_report, should_retrain",
    [
        ({"any_drift": True}, True),
        ({"any_drift": False}, False),
    ],
)
def test_drift_detection(drift_report, should_retrain):
    mock_comparison_data = MagicMock()
    mock_reference_data = MagicMock()

    with patch(
        "databricks_production_ml_system.mlops_service.drift_detection.get_drift_data",
        return_value=(mock_comparison_data, mock_reference_data),
    ) as mock_get_drift_data, patch(
        "databricks_production_ml_system.mlops_service.drift_detection.create_drift_report",
        return_value=drift_report,
    ) as mock_create_drift_report, patch(
        "databricks_production_ml_system.mlops_service.drift_detection.TrainingPipeline.train_and_register_model"
    ) as mock_train_and_register_model:

        result = drift_detection()

        mock_get_drift_data.assert_called_once_with(BEGINNING, MID)

        mock_create_drift_report.assert_called_once_with(
            reference=mock_reference_data, comparison=mock_comparison_data
        )

        if should_retrain:
            mock_train_and_register_model.assert_called_once()
        else:
            mock_train_and_register_model.assert_not_called()

        assert result == drift_report

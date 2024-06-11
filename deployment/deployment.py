from databricks_production_ml_system.utils.constants import (
    DATA_ENGINEERING_PIPELINE_NAME,
    DATA_ENGINEERING_WORKFLOW_CONFIGS,
    DRIFT_DETECTION_PIPELINE_NAME,
    DRIFT_DETECTION_WORKFLOW_CONFIGS,
    MODEL_SERVING_PIPELINE_NAME,
    MODEL_SERVING_WORKFLOW_CONFIGS,
    MODEL_TRAINING_PIPELINE_NAME,
    MODEL_TRAINING_WORKFLOW_CONFIGS,
)
from databricks_production_ml_system.utils.helpers import create_workflow

if __name__ == "__main__":
    create_workflow(
        workflow_configs=DATA_ENGINEERING_WORKFLOW_CONFIGS,
        pipeline_name=DATA_ENGINEERING_PIPELINE_NAME,
    )
    create_workflow(
        workflow_configs=MODEL_SERVING_WORKFLOW_CONFIGS,
        pipeline_name=MODEL_SERVING_PIPELINE_NAME,
    )
    create_workflow(
        workflow_configs=MODEL_TRAINING_WORKFLOW_CONFIGS,
        pipeline_name=MODEL_TRAINING_PIPELINE_NAME,
    )
    create_workflow(
        workflow_configs=DRIFT_DETECTION_WORKFLOW_CONFIGS,
        pipeline_name=DRIFT_DETECTION_PIPELINE_NAME,
    )

from databricks_production_ml_system.mlops_service.drift_detection import (
    drift_detection,
)

if __name__ == "__main__":
    report = drift_detection()

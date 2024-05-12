from databricks_production_ml_system.machine_learning_service.serving_pipeline import (
    ServingPipeline,
)

if __name__ == "__main__":
    ServingPipeline().serve_predictions()

from databricks_production_ml_system.machine_learning_service.training_pipeline import (
    TrainingPipeline,
)

if __name__ == "__main__":
    TrainingPipeline().train_and_register_model()

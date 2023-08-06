from datetime import datetime, timedelta

TODAY = datetime.now()
CUTOFF_TRAIN = TODAY - timedelta(days=730)
CUTOFF_TRAIN = CUTOFF_TRAIN.strftime("%Y-%m-%d")
CUTOFF_EVAL = TODAY - timedelta(days=32)
CUTOFF_EVAL = CUTOFF_EVAL.strftime("%Y-%m-%d")
SCHEMA_TRAINING = "offline"
TABLE_TRAINING = "training"
MODEL_TRAINING_QUERY = f"SELECT * FROM {SCHEMA_TRAINING}.{TABLE_TRAINING} WHERE Date >= {CUTOFF_TRAIN}"
PERFORMANCE_EVAL_QUERY = f"SELECT * FROM {SCHEMA_TRAINING}.{TABLE_TRAINING} WHERE Date >= {CUTOFF_EVAL}"
SCHEMA_SERVING= "online"
TABLE_SERVING = "serving"
MODEL_SERVING_QUERY = f"SELECT * FROM {SCHEMA_SERVING}.{TABLE_SERVING}"
TARGET_COL = "Target"
CATEGORICAL_COLUMNS = ["Feature4"]
COLS_FOR_REMOVAL = ["CustomerState"]
HYPERPARAMS = {
    "penalty": "l2",
    "dual": False,
    "tol": 1e-4,
    "C": 1.0,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "class_weights": None,
    "random_state": 123,
    "solver": "lbfgs",
    "max_iter": 100,
    "multi_class": "auto",
    "verbose": 0,
    "warm_start": False,
    "n_jobs": None,
    "l1_ratio": None,
}
EXPERIMENT_NAME = ""
RUN_NAME = ""
MODEL_NAME = ""
USER = ""
MLFLOW_PROD_ENV = "Production"
PREDICTION_COLS = ["ID", "Prediction", "Prediction_Date"]
THRESHOLD_RETRAIN = 0.6
PREDICTIONS_PATH = "/example_classifier_predictions/*.csv"
PERFORMANCE_EVAL_COLS = ["ID", "Target"]
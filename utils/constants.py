from databricks.feature_store.online_store_spec import AzureSqlServerSpec
from datetime import datetime, timedelta

STORAGE = ""
CONTAINER = ""
MOUNT_NAME = ""
STORAGE_ACC_KEY = ""
EXPERIMENT_NAME = ""
RUN_NAME = ""
MODEL_NAME = ""
USER = ""
FILEPATH = ""
DATE_COL = "Date"
DELIMITER = "\u0001"
MLFLOW_PROD_ENV = "Production"
HTML_DIRECTORY = "dbfs:/FileStore/tmp/index.html"
PREDICTIONS_PATH = "/example_classifier_predictions/*.csv"
TARGET = "Target"
NUMERICAL_COLS = ["Feature1", "Feature2", "Feature3"]
CATEGORICAL_COLS = ["Feature4"]
COLS_FOR_REMOVAL = ["CustomerState"]
PREDICTION_COLS = ["ID", "Prediction", "Prediction_Date"]
PERFORMANCE_EVAL_COLS = ["ID", "Target"]
OFFLINE_TABLE_KEYS = ["ID", "Date", "CustomerNumber"]
OFFLINE_TABLE_PARTITION = ["CustomerState"]
OFFLINE_TABLE_TRAINING_COLS = [
    "ID",
    "Date",
    "CustomerNumber",
    "CustomerState",
    "Feature1",
    "Feature2",
    "Feature3",
    "Feature4",
    "Target",
]
ONLINE_TABLE_KEYS = ["CustomerNumber"]
ONLINE_TABLE_PARTITION = ["CustomerState"]
D_TIME = "Date"
SCHEMA_TRAINING = "offline"
TABLE_TRAINING = "training"
SCHEMA_SERVING = "online"
TABLE_SERVING = "serving"
OFFLINE_TABLE_SCHEMA = "offline"
OFFLINE_TABLE_SERVING = "serving"
OFFLINE_TABLE_TRAINING = "training"
ONLINE_TABLE_SCHEMA = "online"
ONLINE_TABLE = "serving"
TODAY = datetime.now()
CUTOFF_TRAIN = TODAY - timedelta(days=730)
CUTOFF_TRAIN = CUTOFF_TRAIN.strftime("%Y-%m-%d")
CUTOFF_EVAL = TODAY - timedelta(days=32)
CUTOFF_EVAL = CUTOFF_EVAL.strftime("%Y-%m-%d")
CUTOFF = TODAY - timedelta(days=2)
CUTOFF = CUTOFF.strftime("%Y-%m-%d")
BEGINNING = TODAY - timedelta(days=14)
MID = TODAY - timedelta(days=7)
MODEL_TRAINING_QUERY = (
    f"SELECT * FROM {SCHEMA_TRAINING}.{TABLE_TRAINING} WHERE Date >= {CUTOFF_TRAIN}"
)
PERFORMANCE_EVAL_QUERY = (
    f"SELECT * FROM {SCHEMA_TRAINING}.{TABLE_TRAINING} WHERE Date >= {CUTOFF_EVAL}"
)
MODEL_SERVING_QUERY = f"SELECT * FROM {SCHEMA_SERVING}.{TABLE_SERVING}"
ONLINE_TABLE_QUERY = f"SELECT * FROM {OFFLINE_TABLE_SCHEMA}.{OFFLINE_TABLE_SERVING}"
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
OFFLINE_TABLE_DESCRIPTION_SERVING = "Model Serving Feature Table"
OFFLINE_TABLE_DESCRIPTION_TRAINING = "ARDS Feature Table"
OFFINE_TABLE_SERVING_DESCRIPTION = "Model Serving Feature Table"
OFFLINE_TABLE_TRAINING_DESCRIPTION = "ARDS Feature Table"
ONLINE_TABLE_DESCRIPTION = "MRDS Online Feature Table"
THRESHOLD_RETRAIN = 0.6

HOSTNAME = "xxxx.database.windows.net"
PORT = 1433
DATABASE_NAME = ""
ONLINE_TABLE_NAME = "online_feature_table"
READ_SECRET_PREFIX = "kv/online-feature-table"
WRITE_SECRET_PREFIX = "kv/online-feature-table"
ONLINE_STORE = AzureSqlServerSpec(
    hostname=HOSTNAME,
    port=PORT,
    database_name=DATABASE_NAME,
    table_name=ONLINE_TABLE_NAME,
    read_secret_prefix=READ_SECRET_PREFIX,
    write_secret_prefix=WRITE_SECRET_PREFIX,
)

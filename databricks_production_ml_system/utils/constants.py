from datetime import datetime, timedelta

from databricks.feature_store.online_store_spec import AzureSqlServerSpec

STORAGE = ""
CONTAINER = ""
MOUNT_NAME = ""
STORAGE_ACC_KEY = ""
EXPERIMENT_NAME = ""
RUN_NAME = ""
MODEL_NAME = ""
USER = ""
FILEPATH = ""
DATE_COL = "date"
MLFLOW_PROD_ENV = "Production"
TARGET_COL = "target"
PREDICTION_DATE = "prediction_date"
EXPECTATION_SUITE_NAME = "example_expectations"
CHECKPOINT_NAME = "example_checkpoint_name"
NUMERICAL_COLS = ["feature1", "feature2", "feature3"]
CATEGORICAL_COLS = ["feature4"]
COLS_FOR_REMOVAL = ["customer_number"]
PREDICTION_COLS = ["prediction", PREDICTION_DATE]
PERFORMANCE_EVAL_COLS = ["date", "target"]
OFFLINE_TABLE_KEYS = ["date", "customer_number"]
OFFLINE_TABLE_PARTITION = ["date"]
OFFLINE_TABLE_TRAINING_COLS = [
    "customer_number",
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "date",
    "target",
]
ONLINE_TABLE_KEYS = ["customer_number"]
ONLINE_TABLE_PARTITION = ["date"]
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

DATA_SOURCE_CONFIG = {
    "name": "example",
    "class_name": "Datasource",
    "execution_engine": {"class_name": "SparkDFExecutionEngine"},
    "data_connectors": {
        "example_connector": {
            "module_name": "great_expectations.datasource.data_connector",
            "class_name": "RuntimeDataConnector",
            "batch_identifiers": [
                "run_id",
            ],
        }
    },
}

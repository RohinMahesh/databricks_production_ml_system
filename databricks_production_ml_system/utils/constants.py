from datetime import datetime, timedelta

from databricks.feature_store.online_store_spec import AzureSqlServerSpec

# blob storage information
STORAGE = "..."
CONTAINER = "..."
MOUNT_NAME = "..."
STORAGE_ACC_KEY = "..."
EXPERIMENT_NAME = "..."
RUN_NAME = "..."
MODEL_NAME = "..."
USER = "..."

# column names
DATA_FORMAT = "parquet"
DATE_COL = "date"
CUSTOMER_COL = "customer_number"
MLFLOW_PROD_ENV = "Production"
TARGET_COL = "target"
ROW_NUMBER_COLUMN = "row_number"
PREDICTION_DATE = "prediction_date"
NUMERICAL_COLS = ["feature1", "feature2", "feature3"]
CATEGORICAL_COLS = ["feature4"]
COLS_FOR_REMOVAL = [CUSTOMER_COL]
PREDICTION_COLS = ["prediction", PREDICTION_DATE]
PERFORMANCE_EVAL_COLS = [DATE_COL, TARGET_COL]
OFFLINE_TABLE_KEYS = [DATE_COL, CUSTOMER_COL]
OFFLINE_TABLE_PARTITION = [DATE_COL]
OFFLINE_TABLE_TRAINING_COLS = [
    CUSTOMER_COL,
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    DATE_COL,
    TARGET_COL,
]

# dlt and feature table information
DLT_TABLE_NAME = "activity"
ARDS_TABLE_NAME = f"{DLT_TABLE_NAME}_silver"
BRONZE_COMMENT = f"Bronze table for {DLT_TABLE_NAME} data"
SILVER_COMMENT = f"Silver table for {DLT_TABLE_NAME} data"
ONLINE_TABLE_KEYS = [CUSTOMER_COL]
ONLINE_TABLE_PARTITION = [DATE_COL]
SCHEMA_TRAINING = "offline"
TABLE_TRAINING = "training"
SCHEMA_SERVING = "online"
TABLE_SERVING = "serving"
OFFLINE_TABLE_SCHEMA = "offline"
OFFLINE_TABLE_SERVING = "serving"
OFFLINE_TABLE_TRAINING = "training"
ONLINE_TABLE_SCHEMA = "online"
ONLINE_TABLE = "serving"
RESCUED_DATA_COLUMN = "_rescued_data"

# time based filters
TODAY = datetime.now()
CUTOFF_TRAIN = TODAY - timedelta(days=730)
CUTOFF_TRAIN = CUTOFF_TRAIN.strftime("%Y-%m-%d")
CUTOFF_EVAL = TODAY - timedelta(days=32)
CUTOFF_EVAL = CUTOFF_EVAL.strftime("%Y-%m-%d")
CUTOFF = TODAY - timedelta(days=2)
CUTOFF = CUTOFF.strftime("%Y-%m-%d")
BEGINNING = TODAY - timedelta(days=14)
MID = TODAY - timedelta(days=7)

# queries
MODEL_TRAINING_QUERY = (
    f"SELECT * FROM {SCHEMA_TRAINING}.{TABLE_TRAINING} WHERE Date >= {CUTOFF_TRAIN}"
)
PERFORMANCE_EVAL_QUERY = (
    f"SELECT * FROM {SCHEMA_TRAINING}.{TABLE_TRAINING} WHERE Date >= {CUTOFF_EVAL}"
)
MODEL_SERVING_QUERY = f"SELECT * FROM {SCHEMA_SERVING}.{TABLE_SERVING}"
ONLINE_TABLE_QUERY = f"SELECT * FROM {OFFLINE_TABLE_SCHEMA}.{OFFLINE_TABLE_SERVING}"

# ml model parameters
HYPERPARAMS = {
    "penalty": "l2",
    "dual": False,
    "tol": 1e-4,
    "C": 1.0,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "class_weight": None,
    "random_state": 123,
    "solver": "lbfgs",
    "max_iter": 100,
    "multi_class": "auto",
    "verbose": 0,
    "warm_start": False,
    "n_jobs": None,
    "l1_ratio": None,
}
THRESHOLD_RETRAIN = 0.6

# feature table descriptions
OFFLINE_TABLE_DESCRIPTION_SERVING = "Model Serving Feature Table"
OFFLINE_TABLE_DESCRIPTION_TRAINING = "ARDS Feature Table"
OFFINE_TABLE_SERVING_DESCRIPTION = "Model Serving Feature Table"
OFFLINE_TABLE_TRAINING_DESCRIPTION = "ARDS Feature Table"
ONLINE_TABLE_DESCRIPTION = "MRDS Online Feature Table"


# online feature store configs
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

# databricks configs
HOST = "..."
TOKEN = "..."
URL = f"{HOST}/api/2.0/jobs/create"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

SCRIPTS_DIR = "/Workspace/Repos/databricks_production_ml_system/deployment/run_scripts"
REQUIREMENTS_TXT_PATH = (
    "/Workspace/Repos/databricks_production_ml_system/deployment/requirements.txt"
)
DATA_ENGINEERING_PIPELINE_NAME = "Data Engineering Pipeline"
MODEL_SERVING_PIPELINE_NAME = "Model Serving Pipeline"
MODEL_TRAINING_PIPELINE_NAME = "Model Training Pipeline"
DRIFT_DETECTION_PIPELINE_NAME = "Drift Detection Pipeline"

DATA_ENGINEERING_WORKFLOW_CONFIGS = {
    "name": DATA_ENGINEERING_PIPELINE_NAME,
    "new_cluster": {
        "spark_version": "7.3.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 2,
        "spark_conf": {"spark.speculation": False},
        "library_requirements": {"filename": REQUIREMENTS_TXT_PATH},
    },
    "email_notifications": {
        "on_failure": ["rohinmahesh@company.com"],
    },
    "timeout_seconds": 3600,
    "max_retries": 1,
    "schedule": {
        "quartz_cron_expression": "0 8 * * *",  # daily at 8 AM
        "timezone_id": "America/Chicago",
    },
    "tasks": [
        {
            "task_key": "01_create_data",
            "description": f"Run 01_create_data",
            "spark_python_task": {
                "python_file": f"{SCRIPTS_DIR}/01_create_data.py",
            },
        },
        {
            "task_key": "02_create_dlt",
            "description": f"Run 02_create_dlt",
            "spark_python_task": {
                "python_file": f"{SCRIPTS_DIR}/02_create_dlt.py",
            },
        },
        {
            "task_key": "03_create_offline_table",
            "description": f"Run 03_create_offline_training_table",
            "spark_python_task": {
                "python_file": f"{SCRIPTS_DIR}/03_create_offline_training_table.py",
            },
        },
        {
            "task_key": "04_create_serving_table",
            "description": f"Run 04_create_online_serving_table",
            "spark_python_task": {
                "python_file": f"{SCRIPTS_DIR}/04_create_online_serving_table.py",
            },
        },
        {
            "task_key": "05_create_online_table",
            "description": f"Run 05_create_online_table",
            "spark_python_task": {
                "python_file": f"{SCRIPTS_DIR}/04_create_online_table.py",
            },
        },
    ],
}

MODEL_SERVING_WORKFLOW_CONFIGS = {
    "name": MODEL_SERVING_PIPELINE_NAME,
    "new_cluster": {
        "spark_version": "7.3.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 2,
        "spark_conf": {"spark.speculation": False},
        "library_requirements": {"filename": REQUIREMENTS_TXT_PATH},
    },
    "email_notifications": {
        "on_failure": ["rohinmahesh@company.com"],
    },
    "timeout_seconds": 3600,
    "max_retries": 1,
    "schedule": {
        "quartz_cron_expression": "0 11 * * *",  # Daily at 11 AM
        "timezone_id": "America/Chicago",
    },
    "tasks": [
        {
            "task_key": "11_serve_predictions",
            "description": f"Run 11_serve_predictions",
            "spark_python_task": {
                "python_file": f"{SCRIPTS_DIR}/11_serve_predictions.py"
            },
        },
    ],
}

MODEL_TRAINING_WORKFLOW_CONFIGS = {
    "name": MODEL_TRAINING_PIPELINE_NAME,
    "new_cluster": {
        "spark_version": "7.3.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 2,
        "spark_conf": {"spark.speculation": False},
        "library_requirements": {"filename": REQUIREMENTS_TXT_PATH},
    },
    "email_notifications": {
        "on_failure": ["rohinmahesh@company.com"],
    },
    "timeout_seconds": 3600,
    "max_retries": 1,
    "schedule": {
        "quartz_cron_expression": "0 5 1-7 * 1",  # first monday of month at 5 am
        "timezone_id": "America/Chicago",
    },
    "tasks": [
        {
            "task_key": "21_train_model",
            "description": f"Run 21_train_model",
            "spark_python_task": {
                "python_file": f"{SCRIPTS_DIR}/21_train_model.py",
            },
        },
    ],
}

DRIFT_DETECTION_WORKFLOW_CONFIGS = {
    "name": DRIFT_DETECTION_PIPELINE_NAME,
    "new_cluster": {
        "spark_version": "7.3.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 2,
        "spark_conf": {"spark.speculation": False},
        "library_requirements": {"filename": REQUIREMENTS_TXT_PATH},
    },
    "email_notifications": {
        "on_failure": ["rohinmahesh@company.com"],
    },
    "timeout_seconds": 3600,
    "max_retries": 1,
    "schedule": {
        "quartz_cron_expression": "0 4 15-21 * * 2",  # second tuesday of month at 4 am
        "timezone_id": "America/Chicago",
    },
    "tasks": [
        {
            "task_key": "31_check_drift",
            "description": f"Run 31_check_drift",
            "spark_python_task": {
                "python_file": f"{SCRIPTS_DIR}/31_check_drift.py",
            },
        },
    ],
}

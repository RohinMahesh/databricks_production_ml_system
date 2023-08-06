from databricks.feature_store.online_store_spec import AzureSqlServerSpec
from datetime import datetime, timedelta

TODAY = datetime.now()
CUTOFF = TODAY - timedelta(days=2)
CUTOFF = CUTOFF.strftime("%Y-%m-%d")
DATE_COL = "Date"
DELIMITER = "\u0001"
FILEPATH = ""
OFFLINE_TABLE_DESCRIPTION_SERVING = "Model Serving Feature Table"
OFFLINE_TABLE_DESCRIPTION_TRAINING = "ARDS Feature Table"
OFFLINE_TABLE_KEYS = ["ID", "Date", "CustomerNumber"]
OFFLINE_TABLE_PARTITION = ["CustomerState"]
OFFLINE_TABLE_SCHEMA = "offline"
OFFLINE_TABLE_SERVING = "serving"
OFFLINE_TABLE_TRAINING = "training"
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
OFFINE_TABLE_SERVING_DESCRIPTION = "Model Serving Feature Table"
OFFLINE_TABLE_TRAINING_DESCRIPTION = "ARDS Feature Table"
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
ONLINE_TABLE_QUERY = f"SELECT * FROM {OFFLINE_TABLE_SCHEMA}.{OFFLINE_TABLE_SERVING}"
ONLINE_TABLE_DESCRIPTION = "MRDS Online Feature Table"
ONLINE_TABLE_KEYS = ["CustomerNumber"]
ONLINE_TABLE_PARTITION = ["CustomerState"]
ONLINE_TABLE_SCHEMA = "online"
ONLINE_TABLE = "serving"
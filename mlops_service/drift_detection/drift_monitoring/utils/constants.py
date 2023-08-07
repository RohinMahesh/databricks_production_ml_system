from datetime import datetime, timedelta

STORAGE = ""
CONTAINER = ""
MOUNT_NAME = ""
STORAGE_ACC_KEY = ""
HTML_DIRECTORY = "dbfs:/FileStore/tmp/index.html"
TARGET = "Target"
NUMERICAL_COLS = ["Feature1", "Feature2", "Feature3"]
CATEGORICAL_COLS = ["Feature4"]
D_TIME = "Date"
TODAY = datetime.now()
BEGINNING = TODAY - timedelta(days=14)
MID = TODAY - timedelta(days=7)

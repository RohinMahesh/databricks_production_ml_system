from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from datetime import date
from helperFunctions import trigger_actions
import pandas as pd
from sklearn.metrics import accuracy_score

# Load predictions
today = datetime.now()
cutoff = today - timedelta(days=32)
cutoff = cutoff.strftime("%Y-%m-%d")

path = "/example_classifier_predictions/*.csv"
predictions = spark.read.options(header=True).csv(path)
predictions = predictions.filter(predictions.Prediction_Date >= cutoff)

# Load labels
fs = feature_store.FeatureStoreClient()
query = "SELECT * FROM offline.training WHERE Date >= {0}".format(cutoff)
data = spark.sql(query)
data = data.select("ID", "Target")
to_evaluate = data.join(predictions, data.ID == predictions.ID, "left").toPandas()

# Calculate performance
performance = accuracy_score(
    list(to_evaluate["Target"]), list(to_evaluate["Prediction"])
)

# Trigger actions
trigger_actions(score=performance, acceptable_performance=0.6)
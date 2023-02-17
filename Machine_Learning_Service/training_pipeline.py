from databricks import feature_store
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from MLflowFunctions import register_mlflow
import pyspark.sql.functions as func
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Load data
today = datetime.now()
cutoff = today - timedelta(days=730)
cutoff = cutoff.strftime("%Y-%m-%d")

fs = feature_store.FeatureStoreClient()
query = "SELECT * FROM example.training WHERE Date >= {0}".format(cutoff)
data = spark.sql(query)

# Engineer features
data = data.groupby("CustomerNumber").agg(
    func.min("Feature1").alias("Feature1"),
    func.max("Feature2").alias("Feature2"),
    func.sum("Feature3").alias("Feature3"),
    func.last("Feature4").alias("Feature4"),
    func.last("Target").alias("Target"),
)

# Convert to Pandas
X_train = data.toPandas()
y_train = X_train["Target"]
del X_train["Target"]


# Define feature extractor
categorical_features = ["Feature4"]
categorical_transformer = Pipeline(
    steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
)
feature_extractor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
)

# Define hyperparameters for logging
hyperparams = {
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

# Fit model
clf = Pipeline(
    steps=[
        ("preprocessor", feature_extractor),
        ("model", LogisticRegression(random_state=123)),
    ]
)
clf.fit(X_train, y_train)

# Register in MLflow
status = register_mlflow(
    experiment_name="sample_experiment",
    run_name="Run1",
    model_name="example_classifier",
    user="Rohin Mahesh",
    data=X_train,
    model=clf,
    parameters=hyperparams,
    stage="Staging",
)

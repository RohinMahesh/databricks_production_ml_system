# Databricks Production ML System

![Screenshot](databricks_production_ml_system/docs/images/architecture_diagram.jpg)

## Background
This repository contains an end-to-end production machine learning system using Azure Databricks. In this system, we are implementing a single deployment strategy for a batch deployment. All relevant downstream data assets, ML artifacts and services are scheduled at a regular cadence.

Below are the core services in this ML system:
1. Data Engineering Service
2. Machine Learning Service
3. MLOps Service

This ML System leverages the following technologies:
1. Delta Live Tables for Data Engineering Service and Data Quality Validation Pipeline
2. Python for ML pipeline development and Drift Detection Pipeline
3. MLflow for model artifact management
4. Feature Store for access to high-quality Analytics and Model Ready Data Sets in low latency
5. EvidentlyAI for Drift Detection

## Data Engineering Service

The Data Engineering Service consists of 3 scheduled pipelines that performs ETL from the upstream data assets into downstream feature tables. In this system, the source-aligned data assets are in a Parquet format and the downstream data assets resulting from these pipelines are stored in a Delta format and in an Azure SQL Server table. 

Details for the Data Engineering Services and its components can be found in the "data_engineering_service" folder: 
1. Delta Live Table: following a Medallion Architecture, this component creates a Delta Live Table (DLT) that takes incremental data from upstream Parquet files, performs different expectations/validations for data quality, and creates a representation for downstream consumption as an Analytics Ready Data Set (ARDS). This is scheduled to run on a daily cadence:
    - create_data.py
    - create_dlt.py
2. Offline Feature Table (FT) Pipeline: populates the offline feature store to create the Analytics Ready Data Set (ARDS) created from the output of the Delta Live Table (DLT). This is updated using the following script and is scheduled to run on a daily cadence:
    - feature_store_training_daily.py
3. Online Feature Table (FT) Pipeline: populates the online feature store to create the Model Ready Data Set (MRDS). This is updated using the following script and is scheduled to run on a daily cadence:
    - feature_store_serving_daily.py
    - online_feature_table_daily.py

## Machine Learning Service

The Machine Learning Service consists of 3 scheduled pipelines that leverages the outputs of the Data Engineering Service to train an ML pipeline and use that pipeline to service predictions for downstream consumption. Model training and serving pipelines utilizes the Databricks MLflow integration for model artifact management and the respective Databricks Feature Store tables. Finally, the overall ML pipeline is evaluated and retrained if model has deteriorated.

Details for the Machine Learning Services and its components can be found in the "machine_learning_service" folder: 
1. Model Training Pipeline: trains ML pipeline, which packs both the feature extractor and ML model into a serialized scikit-learn Pipeline object. This is scheduled to run on a monthly cadence.
2. Model Serving Pipeline: services ML pipeline for predictions. This is scheduled to run on a daily cadence.
3. Model Performance Evaluation Pipeline: evaluates ML pipeline on last 31 days worth of data (accounting for a 1-day label delay) and triggers model retraining based on acceptable performance of ML pipeline.

## MLOps Service

The MLOps Service consists of 1 scheduled pipelines that leverages the outputs of the Machine Learning Service to detect any data drift and subsequently trigger model retraining. This drift detection pipeline will be integrated into downstream CI/CD/CT and alerting functionalities.

Details for the MLOps Services and its components can be found in the "mlops_service" folder: 
1. Drift Detection: drift detection framework to calculate distributional shifts in our ARDS and trigger model retraining. This is scheduled to run on a monthly cadence.
    - drift_detection.py

## Orchestration
This ML system consists of various pipelines that are orchestrated using Databricks Jobs. For full control over your jobs, test converge, and CI/CD/CT, it is recommended to leverage Azure DevOps and the Databricks CLI. It is also recommended to utilize the GitHub Databricks integration for orchestration of the Databricks Jobs/Azure DevOps Pipelines.



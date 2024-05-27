# Data Quality Monitoring
## Motivation

![Screenshot](images/data_engineering_lifecycle.png)

The backbone to successful ML Systems and Data Science products are the foundational data pipelines that takes data from the source and creates a valuable representation to be served to various downstream consumers. These pipelines ensure that the data served to downstream consumers are reliable, available and of high quality. 

A hidden technical debt many Data teams face is not implementing proper monitoring of the data assets used by various downstream consumers, such as dashboards or ML systems. Due to the complexity of many ETL pipelines and the various points of failure in the Data Engineering Lifecycle, it is vital to implement data quality monitoring throughout the lifecycle to validate the data. This ensures that the data served to all downstream consumers are reliable and of high-quality. In this repository, the ETL pipelines are built using Delta Live Tables (DLT). Expectations are created for the data assets, ensuring high-quality and reliable data for downstream machine learning solutions. 

![Screenshot](images/evidently_monitoring.png)
When monitoring production ML systems, it is imperative we monitor:
- Data health: monitoring of our upstream data assets (ex: schema changes, broken pipelines, etc.)
- Model health: monitoring of different ML components (ex: concept/covariate drift etc.)
- Service health: monitoring of overall resources and infrastructure

In this service, we focus on building robust and scalable ETL pipelines to transform raw source-aligned data into analytics and model ready data sets. We monitoring the data health by leveraging Delta Live Tables for expectation validation and utilize Databricks Feature Store for storing and serving analytics and model ready data sets.

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

## References
EvidentlyAI, https://www.evidentlyai.com, 2022

Reis, Joe, and Matt Housely. Fundamentals of Data Engineering: Plan and Build Robust Data 
    Systems. O'Reilly, 2022.

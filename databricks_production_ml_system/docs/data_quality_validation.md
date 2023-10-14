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

In this project, we focus on monitoring the Data health by leveraging Delta Live Tables for expectation validation. 

## References
EvidentlyAI, https://www.evidentlyai.com, 2022

Reis, Joe, and Matt Housely. Fundamentals of Data Engineering: Plan and Build Robust Data 
    Systems. O'Reilly, 2022.

# Machine Learning Service

## Background

In this service, we utilize the Model Ready Data Set (MRDS) in the Feature Store that is populated by the Data Engineering service to train our ML model and utilize the Online Feature Store for serving predictions. 

As we construct a cost and benefit metrix of each incorrect and correct predictions, in our use case False Negatives have a higher cost for incorrect predictions in comparison to False Positives. Resulting from this select the F1-Score as our evaluation metric as we are still interested in controlling for False Positives despite False Negatives having a higher cost. This is crucial for model selection and downstream hyperparameter tuning.

Given our sample size and feature space, the Logistic Regression yielded the best performance across our training, validation, and testing sets. Additionally, when creating a learning curve we found that we were not getting a lift in the F1-Score after 2 years worth of data. Resulting from this, when triggering the Model Training Pipeline to update the coefficients with new fresh data, we only select the last 2 years worth of data as every customer has at least 1 interaction a day and we have a 1 day label delay.

This Logistic Regression model is packaged into a scikit-learn pipeline with a OneHotEncoder to encode nominal level categorical variables. This pipeline is then registered and served using MLflow to provide proper governance around our ML model artifacts.

## Design

The Machine Learning Service consists of 3 scheduled pipelines that leverages the outputs of the Data Engineering Service to train an ML pipeline and use that pipeline to service predictions for downstream consumption. Model training and serving pipelines utilizes the Databricks MLflow integration for model artifact management and the respective Databricks Feature Store tables. Finally, the overall ML pipeline is evaluated and retrained if model has deteriorated.

Details for the Machine Learning Services and its components can be found in the "machine_learning_service" folder: 
1. Model Training Pipeline: trains ML pipeline, which packs both the feature extractor and ML model into a serialized scikit-learn Pipeline object. This is scheduled to run on a monthly cadence.
2. Model Serving Pipeline: services ML pipeline for predictions. This is scheduled to run on a daily cadence.
3. Model Performance Evaluation Pipeline: evaluates ML pipeline on last 31 days worth of data (accounting for a 1-day label delay) and triggers model retraining based on acceptable performance of ML pipeline.
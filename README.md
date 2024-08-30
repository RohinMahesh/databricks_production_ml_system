# Databricks Production ML System
![Build Status](https://github.com/RohinMahesh/databricks_production_ml_system/actions/workflows/ci.yml/badge.svg)

![Screenshot](databricks_production_ml_system/docs/images/architecture_diagram.jpg)

## Background
This repository contains an end-to-end production machine learning system using Azure Databricks. In this system, we are implementing a single deployment strategy for a batch deployment. All relevant downstream data assets, ML artifacts, and services are orchestrated using Databricks Workflows.

Below are the core services in this ML system:
1. Data Engineering Service
2. Machine Learning Service
3. MLOps Service

This ML System leverages the following technologies:
1. Delta Live Tables for Data Engineering Service and Data Quality Validation Pipeline
2. Python and scikit-learn for ML pipeline development and Drift Detection Pipeline
3. MLflow for model artifact management
4. Databricks Feature Store for access to high-quality Analytics and Model Ready Data Sets in low latency
5. EvidentlyAI for Drift Detection
6. Databricks Workflows for orchestration of the ML System

Under the README of the individual services, you can find more details on the components and functionalities that are supported.
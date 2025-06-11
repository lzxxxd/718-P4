# Spaceship Titanic Prediction

This repository contains my submission for **Project 04 - Models** in AAE 718: Data Science for Agricultural and Applied Economics (Summer 2025).

## 🔍 Project Overview

This project tackles the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/) prediction challenge hosted on Kaggle. The goal is to predict whether passengers were transported to an alternate dimension based on various demographic and spending behavior data.

The core task involves:
- Data cleaning and feature engineering
- Model training using scikit-learn pipelines
- Evaluation and model comparison
- Generating predictions on unseen test data

## 🧠 Model Used

The final model selected is:
- `HistGradientBoostingClassifier` from `sklearn.ensemble`

It achieved a **validation accuracy of 83.05%**, significantly exceeding the required benchmark of 75%.

Other models tested:
- Random Forest
- Gradient Boosting
- XGBoost

## 🛠 Technologies

- Python 3
- scikit-learn
- pandas
- numpy
- seaborn / matplotlib (for visualization)
- Jupyter Notebook (optional, not included here)

## 📁 Repository Structure

spaceship_model_hgb.py # Final Python script with full pipeline
submission_hgb.csv # Kaggle submission file (generated after script run)
README.md # Project overview and instructions (this file)
.gitignore # Prevents uploading large data/API keys

## 🔐 Notes
No .csv data files are included in this repository as per instruction.

No API keys were required for this project.


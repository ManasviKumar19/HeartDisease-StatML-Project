# Heart Disease Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-brightgreen.svg)]()
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)]()
[![Status](https://img.shields.io/badge/Project-Completed-success.svg)]()

This repository contains an end-to-end Machine Learning pipeline to predict **Heart Disease**, using the **UCI Heart Dataset**.  
The project includes data preprocessing, EDA, model training, hyperparameter tuning, and evaluation across multiple ML models.

---

## 📁 Repository Contents

| File | Description |
|---|---|
`heart_disease_ml_analysis.py` | Full ML pipeline script |
`Heart_Disease_ML_Analysis.ipynb` | Interactive notebook with plots |
`Heart_Disease_Prediction_Presentation.pptx` | Presentation summarizing insights |
`README.md` | Project overview |

---

## 🎯 Objective

- Perform exploratory data analysis (EDA)
- Preprocess clinical heart disease data
- Train multiple ML models
- Tune hyperparameters
- Compare performance & interpret important features

---

## 🧠 Dataset

**Source:** UCI Machine Learning Repository  
**Features:** Age, Sex, Chest Pain, Cholesterol, Thalassemia, etc.  
**Target:** Heart Disease Presence (0 = No, 1 = Yes)

---

## 📊 Exploratory Data Analysis

Key visualizations include:

✅ Age & cholesterol distribution  
✅ Gender-wise disease pie charts  
✅ Feature correlations heatmap  
✅ Pairplots of critical variables  
✅ Boxplots comparing disease vs health indicators  

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
Logistic Regression | Baseline + GridSearch tuning |
Random Forest | Best accuracy + feature importance |
Support Vector Machine (SVC) | Linear kernel + tuning |
K-Means Clustering | Exploratory pattern analysis |

---

## 🏆 Performance Summary

| Model | Accuracy |
|---|---|
Random Forest | ⭐ ~82–85% |
Logistic Regression | ~77–80% |
SVM | ~75–80% |

**Top predictors:** `thalach`, `oldpeak`, `chol`, `age`, `ca`

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python heart_disease_ml_analysis.py

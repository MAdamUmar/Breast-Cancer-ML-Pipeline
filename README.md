# Breast Cancer Prediction with Machine Learning

An end-to-end machine learning pipeline for breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer dataset. This project features complete data preprocessing, feature selection, model comparison, hyperparameter tuning, and detailed performance evaluation through visuals.

Achieved a final accuracy score of 97% and a final F1 score of 0.965 and deployed it as a web app using Flask.

---

## 📂 Project Structure

- 📂 model folder:
    - pipeline.pkl (The trained ML pipeline file, saved with joblib for the Flask app)
- 📂 templates folder
    - index.html (The frontend form, written in HTML)
- app.py (The main backend script, the brain of the web app)
- requirements.txt (Dependecies to be installed)
- Complete ML Workflow Notebook.ipynb (Complete notebook with code, models, and evaluation)
- data.csv (Dataset used in the notebook, source is [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) )
- Final Model Performance.png (Visual: Best model's scores and confusion matrix)
- F1 Score Model Comparison.png (Visual: F1 score comparison across models)


---

## 📊 Models Used

- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **K-Nearest Neighbors (KNN)**
- **Multi-layer Perceptron (MLP)**
- **Random Forest**
- **XGBoost**

GridSearchCV was used to optimize hyperparameters for the top-performing models.

---

## 🧪 Evaluation Metrics

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📈 Visualizations

Key insights and comparisons include:
- Feature importance via mutual information
- F1 score bar chart
- Final confusion matrix of the best model

<p align="center">
  <img src="F1 Score Model Comparison.png" width="400"/> 
  <img src="Final Model Performance.png" width="400"/>
</p>

---

## 🛠 Tech Stack

- Python
- scikit-learn
- XGBoost
- pandas, numpy, matplotlib, seaborn
- Jupyter Notebook
- Flask
- Joblib

---

## 🧩 Features

- Clean preprocessing and scaling pipeline to analyze and predict medical data
- Feature selection based on mutual information and visualisation using python
- Comparison of multiple ML classifiers 
- Hyperparameter tuning using GridSearchCV
- Dealing with class imbalance using SMOTE
- Voting Classifier (ensemble model of XGBClassifier and RandomForestClassifier)
- Professional-level presentation and reproducible notebook

---

## 🎯 Goal

To build a reliable, interpretable, and deployable classification pipeline for breast cancer detection and deploying the pipeline as a Web App.

## Author
Muhammad Adam Umar
Connect on:
[Linkedin](https://www.linkedin.com/in/muhammad-adam-umar-26baaa2b5/)
[Github](https://github.com/MAdamUmar/)

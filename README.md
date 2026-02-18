# ❤️ ECG / EKG Signal Classification & Disease Prediction

> End-to-end machine learning pipeline for ECG-based disease classification using structured clinical data.

This project explores predictive modelling on ECG/EKG datasets to classify cardiovascular disease labels using:

- Statistical preprocessing  
- Feature encoding & scaling  
- Class imbalance handling (SMOTE)  
- Cross-validated model benchmarking  
- Hyperparameter optimization  

The goal was to rigorously evaluate multiple ML classifiers and identify the most robust predictive model.

---

## 📌 Problem Statement

Given structured ECG-derived and clinical features, the objective is to:

- Predict disease classification labels  
- Handle imbalanced class distributions  
- Compare baseline and ensemble methods  
- Evaluate generalisation via stratified cross-validation  

---

## 🏗️ Project Structure

EKG-1.ipynb # Full modelling notebook
dataset.xlsx # ECG dataset (not included in repo)
README.md


---

## 🧠 Pipeline Overview

### 1️⃣ Data Loading
- Excel-based structured dataset
- Numerical + categorical features
- Disease label as target variable

---

### 2️⃣ Exploratory Data Analysis
- Data type inspection
- Missing value detection
- Automated profiling via `ydata-profiling`
- Distribution & feature inspection

---

### 3️⃣ Preprocessing

#### Numerical Features
- StandardScaler normalization

#### Categorical Features
- One-Hot Encoding (drop first to avoid multicollinearity)
- Label Encoding for disease label

#### Final Dataset
- Concatenated scaled numeric + encoded categorical features

---

### 4️⃣ Train/Test Split

- 80/20 split  
- Random state fixed for reproducibility  

---

### 5️⃣ Class Imbalance Handling

Used **SMOTE (Synthetic Minority Over-sampling Technique)**:

- Dynamic `k_neighbors` selection based on smallest class size  
- Applied only when sufficient minority samples exist  
- Prevents model bias toward dominant classes  

---

### 6️⃣ Model Benchmarking

Evaluated multiple classifiers:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- (Configured but optional) Gradient Boosting  
- AdaBoost  
- XGBoost  

---

### 7️⃣ Hyperparameter Tuning

- GridSearchCV  
- 5-fold Stratified Cross Validation  
- Accuracy-based scoring  
- Best estimator selection  

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

StratifiedKFold ensures class distribution is preserved across folds.

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras  
- imbalanced-learn (SMOTE)  
- Seaborn & Matplotlib  
- ydata-profiling  

---

## ⚙️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn seaborn matplotlib ydata-profiling openpyxl tensorflow
2️⃣ Open Notebook
jupyter notebook EKG-1.ipynb
3️⃣ Provide Dataset
Place the dataset file in the correct directory and update the path if necessary:

df = pd.read_excel("NEW DATASET EKG.xlsx")
📈 Potential Improvements
Feature selection via RFE / SelectKBest

PCA dimensionality reduction experiments

Neural network classification benchmarking

ROC-AUC comparison

SHAP-based feature importance interpretation

Full pipeline encapsulation using sklearn.pipeline

⚠️ Notes
Dataset not included due to size/privacy

Notebook designed for experimentation and model comparison

Intended for research & academic modelling purposes

# Cybersecurity Threat Classification Using Machine Learning
📌 Project Overview
This project aims to classify cybersecurity threats based on a given dataset using Machine Learning (ML) techniques. The pipeline includes data preprocessing, feature selection, model training, and evaluation.

## We implement and compare four ML models:
✅ Random Forest
✅ Support Vector Machine (SVM)
✅ K-Nearest Neighbors (KNN)
✅ Neural Network (MLP)

The best-performing model is used to predict threats on the test dataset, and results are saved to a CSV file.

## 📂 Dataset
Train_data.csv: Contains labeled threat data for training the models.

Test_data.csv: Contains unlabeled threat data for classification.

## 🔹Features
Numerical Features: Various network attributes related to cybersecurity threats.

Categorical Features: Encoded service types, protocols, etc.

Target Variable: "class" (Threat label).

## 📊 Data Preprocessing
✔ Handle missing values – Fill numeric columns with the median value.
✔ One-hot encode categorical features – Convert categorical columns into numeric form.
✔ Align test dataset – Ensure train and test datasets have the same feature set.
✔ Feature selection – Use SelectKBest to choose the top 50 most relevant features.
✔ Standardization – Scale features using StandardScaler for better model performance.

##🔹 Performance Metrics Used:

Accuracy: Measures overall correctness.

Precision: Measures how many predicted threats were actually threats.

Recall: Measures how many actual threats were correctly identified.

F1-score: Balances Precision & Recall.

## 📊 Visualization
🔹 Confusion Matrices for each model.
🔹 Feature Importance Plot (Top 10 important features).

## 📌 Results & Predictions
The best-performing model (Random Forest) is used to predict threats on the test dataset.

Predictions are saved in Predicted_Test_Results.csv.

## ⚙️ How to Run the Project
1️⃣ Install dependencies:

```python
pip install pandas numpy matplotlib seaborn scikit-learn
```

2️⃣ Run the Python script:
```python
python cybersecurity_classification.py
```
3️⃣ Check the generated predictions:

The results will be saved as Predicted_Test_Results.csv.

## 📌 Future Enhancements
✅ Hyperparameter tuning for better accuracy.
✅ Use SMOTE to handle class imbalance (if dataset is imbalanced).
✅ Try deep learning models (CNN, LSTM) for better classification.

🚀 Developed for Cybersecurity Threat Detection using ML
📧 Feel free to contribute or suggest improvements! 😊

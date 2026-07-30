# Diabetes Risk Prediction Using Machine Learning
## Live Demo

🔗 https://diabetes-risk-prediction-using-machine-qij4.onrender.com

## Overview
This project predicts whether a patient is likely to have diabetes using Machine Learning models. Users enter medical information through a Flask web application, and the system compares predictions from five different models.

## Features
- Predict diabetes risk from patient data
- Compare 5 Machine Learning models
- Display Accuracy, F1-Score, Recall, and ROC-AUC
- Highlight the best-performing model
- User-friendly web interface using Flask

## Machine Learning Models
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

## Technologies Used
- Python
- Flask
- Scikit-learn
- NumPy
- Joblib
- HTML
- CSS

## Dataset
- Pima Indians Diabetes Dataset

## Project Structure

Diabetes-Prediction/
│── app.py
│── requirements.txt
│── README.md
│── model_DT.pkl
│── model_KNN.pkl
│── model_LR.pkl
│── model_RF.pkl
│── model_SVM.pkl
│── scaler.pkl
├── templates/
│ ├── index.html
│ └── D.html
└── dataset/
├── diabetes.csv
└── Diabetes1.ipynb

## Installation

```bash
pip install -r requirements.txt
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

## Future Improvements

- Deploy online using Render
- Add more datasets
- Improve prediction accuracy
- Add visualization dashboard

## Author

**Md. Najmus Shakib**

Department of Electrical & Electronic Engineering (EEE)

Rajshahi University of Engineering & Technology (RUET)

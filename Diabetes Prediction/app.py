from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained models
model_LR = joblib.load("model_LR.pkl")
model_KNN = joblib.load("model_KNN.pkl")
model_SVM = joblib.load("model_SVM.pkl")
model_DT = joblib.load("model_DT.pkl")
model_RF = joblib.load("model_RF.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    data = [
        float(request.form["Pregnancies"]),
        float(request.form["Glucose"]),
        float(request.form["BloodPressure"]),
        float(request.form["SkinThickness"]),
        float(request.form["Insulin"]),
        float(request.form["BMI"]),
        float(request.form["DPF"]),
        float(request.form["Age"])
    ]

    # Convert into NumPy array
    input_data = np.array(data).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predictions
    pred_LR = model_LR.predict(input_scaled)[0]
    pred_KNN = model_KNN.predict(input_scaled)[0]
    pred_SVM = model_SVM.predict(input_scaled)[0]
    pred_DT = model_DT.predict(input_scaled)[0]
    pred_RF = model_RF.predict(input_scaled)[0]

    # Send results to HTML page
    return render_template(
        "D.html",
        LR_result=pred_LR,
        KNN_result=pred_KNN,
        SVM_result=pred_SVM,
        DT_result=pred_DT,
        RF_result=pred_RF
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

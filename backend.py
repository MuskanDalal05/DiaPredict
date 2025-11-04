from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
df = pd.read_csv("diabetes_type.csv")
X = df.drop(["Outcome", "Diabetes_Type"], axis=1)
y_outcome = df["Outcome"]
y_type = df["Diabetes_Type"]

# Train models
model_outcome = RandomForestClassifier(random_state=42)
model_outcome.fit(X, y_outcome)

model_type = RandomForestClassifier(random_state=42)
model_type.fit(X, y_type)

@app.route("/")
def form():
    return render_template("frontend.html")


# Step 1: Predict diabetic or not
@app.route("/predict", methods=["POST"])
def predict():
    input_data = [[
        int(request.form["Pregnancies"]),
        int(request.form["Glucose"]),
        int(request.form["BloodPressure"]),
        int(request.form["SkinThickness"]),
        int(request.form["Insulin"]),
        float(request.form["BMI"]),
        float(request.form["DiabetesPedigreeFunction"]),
        int(request.form["Age"])
    ]]
    
    outcome_pred = model_outcome.predict(input_data)[0]
    result_text = "Diabetic" if outcome_pred == 1 else "Not Diabetic"

    return render_template("result.html", outcome=result_text, show_type=(outcome_pred==1), input_data=input_data)


# Step 2: Predict type if user wants
@app.route("/check_type", methods=["POST"])
def check_type():
    input_data = eval(request.form["input_data"])  # hidden input from result page
    type_pred = model_type.predict(input_data)[0]
    return render_template("type_result.html", diabetes_type=type_pred)


if __name__ == "__main__":
    app.run(debug=True)

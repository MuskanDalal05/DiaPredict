🩺 Diapredict — Diabetes Prediction Web App

Diapredict is a machine learning–powered web application built using Flask that predicts whether a person is diabetic based on health parameters.
If diagnosed as diabetic, the app further identifies the type of diabetes using a trained Random Forest Classifier.

🚀 Features

Predicts whether a user is Diabetic or Not Diabetic
Identifies Type 1 or Type 2 Diabetes for diabetic users
User-friendly and interactive web interface
Built with Flask, HTML/CSS, and Scikit-learn
Trained on a real-world Diabetes Dataset.

🧠 Tech Stack

Backend: Python (Flask)
Frontend: HTML, CSS
Machine Learning: Scikit-learn (Random Forest)
Dataset: diabetes_type.csv

📂 Project Structure
Diapredict/
│
├── backend.py               # Flask backend and ML model training
├── diabetes_type.csv        # Dataset used for training and testing
├── frontend.html            # Main input form page
├── type_result.html         # Page displaying diabetes type
├── static/                  # Folder for images (e.g., bg.PNG)
└── LICENSE                  # MIT License file

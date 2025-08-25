import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

# Createing an app object
app = Flask(__name__)

# Load the trained model from the file.
model = joblib.load('churn_prediction_pipeline.joblib')


# Route for the home page.
@app.route('/')
def home():
    return render_template('index.html')


# Route for the prediction page.
@app.route('/predict', methods=['POST'])
def predict():

    # Get the input values from the form.
    # The keys must match the 'name' attributes of your form's input fields.
    input_features = [request.form.get(key) for key in model.feature_names_in_]

    # Convert the input to a DataFrame.
    features_df = pd.DataFrame([input_features], columns=model.feature_names_in_)

    # The model expects numerical types for these columns, so we convert them.
    # This is a crucial step if your form sends everything as text.
    numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    # Make a prediction.
    prediction = model.predict(features_df)
    prediction_proba = model.predict_proba(features_df)

    # Determine the output message.
    if prediction[0] == 1:
        output_message = 'This customer is LIKELY to churn.'
        confidence = f"Confidence: {prediction_proba[0][1]:.2%}"
    else:
        output_message = 'This customer is UNLIKELY to churn.'
        confidence = f"Confidence: {prediction_proba[0][0]:.2%}"

    # Render the index.html page with the prediction result.
    return render_template('index.html', prediction_text=output_message, confidence_text=confidence)


# This is the standard boilerplate that runs the application.
if __name__ == "__main__":
    app.run(debug=True)

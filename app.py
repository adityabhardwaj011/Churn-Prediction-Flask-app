import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

# Creating an app object
app = Flask(__name__) # Think of __name__ can be seen as the "home address" we give to Flask. If we give it something else, it won't be able to find its way back to its files. Using the special __name__ variable ensures that Flask always knows the correct location of your app.py script and templates etc. no matter how or where we run it from.
# It's a special, built-in Python variable that holds the name of the current Python script (or module). 
# Load the trained model from the file.
model = joblib.load('churn_prediction_pipeline.joblib')


# Route for the home page.
@app.route('/')
def home():
    return render_template('index.html', form_data= {}) # to display an empty form


# Route for the prediction page.
@app.route('/predict', methods=['POST'])
def predict():

    # Get the input values from the form.
    # The keys must match the 'name' attributes of our form's input fields.
    input_features = [request.form.get(key) for key in model.feature_names_in_]

    # Convert the input into a DataFrame cause that's what the model expects
    features_df = pd.DataFrame([input_features], columns=model.feature_names_in_)

    # data from HTML always comes as TEXT
    # The model expects numerical types for these columns, so we convert them.
    numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    # Making a prediction.
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
    return render_template('index.html', prediction_text=output_message, confidence_text=confidence, form_data=request.form) # to display the form user filled with the result

if __name__ == "__main__": # This line checks, "Is this script being run directly?" If it is, then __name__ will be "__main__", the condition will be true, and the web server will start. This prevents the server from starting automatically if you were to just import this file into another script.
    app.run(debug=True)

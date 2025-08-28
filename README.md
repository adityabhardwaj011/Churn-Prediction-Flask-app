# Telco Customer Churn Prediction App

A web application that predicts customer churn using a machine learning model trained on the Telco Customer Churn dataset. This project takes a raw dataset, preprocesses it, trains a Random Forest Classifier, and deploys it as an interactive web app using Flask.

![Demo Image](./assets/churn_no_churn.gif)

---

## 📋 Features

-   **Interactive UI:** A user-friendly web form to input customer data.
-   **Real-time Predictions:** Instantly predicts whether a customer is likely to churn.
-   **Confidence Score:** Provides the probability of the prediction.
-   **Machine Learning Pipeline:** Uses a Scikit-learn pipeline for robust data preprocessing and prediction.
-   **Imbalanced Data Handling:** Implements SMOTE to handle the class imbalance in the dataset.

---

## 🛠️ Tech Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** Scikit-learn, Pandas, NumPy, Imbalanced-learn
-   **Frontend:** HTML, CSS

---

## 🧠 Model Training

The machine learning model was trained in a Kaggle environment. The complete data exploration, feature engineering, and training process can be found in the `churn-no-churn-prediction.ipynb` notebook in this repository.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need to have Python 3 installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adityabhardwaj011/Churn-Prediction-Flask-app.git
    cd Churn-Prediction-Flask-app
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ Usage

1.  **Download the Dataset:** Make sure you have the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in the root of the project directory.
2.  **Run the Flask application:**
    ```bash
    python app.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:5000` to use the app.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

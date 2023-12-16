

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import joblib
import os

app = Flask(__name__)

# First train and save the "mpg" model
def train_and_save_mpg_model():
    # Load the "mpg" dataset
    data = load_boston()
    X, y = data.data[:, 0:1], data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model using joblib
    joblib.dump(model, 'mpg_model.joblib')

# Load the models
train_and_save_mpg_model()
diabetes_model = joblib.load(r"C:\Users\Charles\Documents\CNA Lectures\Third Term Sept to Dec 2023\Emerging Trends & Innovation CP4477 Arun Rameshbabu\FlaskMachineLearningProject\DiabetesTasks.joblib")
mpg_model = joblib.load("mpg_model.joblib")

# Create the Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from the user
        mpg_input = float(request.form['mpg_input'])
        diabetes_input = float(request.form['diabetes_input'])

        # Make predictions from the model
        mpg_prediction = mpg_model.predict([[mpg_input]])[0]
        diabetes_prediction = diabetes_model.predict([[diabetes_input]])[0]

        return render_template('result.html', mpg_prediction=mpg_prediction, diabetes_prediction=diabetes_prediction)

    except ValueError:
        # Handle error for inputed numbers from users
        return render_template('error.html', error_message="Invalid input. Please enter numeric values.")

if __name__ == '__main__':
    app.run(debug=True)

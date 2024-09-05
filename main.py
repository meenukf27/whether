from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')


# Define the home route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')


# Define the prediction route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    pressure = float(request.form['pressure'])
    summary = int(request.form['summary'])  # Encoded summary value

    # Convert the input data to a DataFrame
    custom_data = pd.DataFrame({
        'Temperature (C)': [temperature],
        'Humidity': [humidity],
        'Pressure (millibars)': [pressure],
        'Summary': [summary]
    })

    # Apply the same scaling as in the training data
    scaler = MinMaxScaler()
    custom_features = scaler.fit_transform(custom_data)

    # Convert the single data point to the expected input shape for the model
    SEQ_LENGTH = 3  # Ensure this matches your training configuration
    X_custom = np.array([custom_features[0]] * SEQ_LENGTH).reshape(1, SEQ_LENGTH, -1)

    # Make the prediction
    prediction = model.predict(X_custom)

    # Convert the prediction to binary output (1 for rain, 0 for no rain)
    predicted_class = (prediction > 0.5).astype(int)[0][0]

    # Return the result to the HTML page
    return render_template('index.html', prediction=predicted_class)


# Run the application
if __name__ == '__main__':
    app.run(debug=False , host='0.0.0.0')

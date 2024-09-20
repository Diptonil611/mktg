import streamlit as st
import pandas as pd
import pickle

# Load the trained model
filename = 'rf_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a title for the app
st.title("Marketing Campaign Response Prediction")

# Create input fields for the features
st.header("Enter Customer Information:")

# Assuming your features are named 'Age', 'Income', 'Education', etc.
# Replace these with your actual feature names
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=50000)
education = st.number_input("Education Level (e.g., 1-5)", min_value=1, max_value=5, value=3)
# ... add more input fields for other features

# Create a button to make predictions
if st.button("Predict Response"):
    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Education': [education],
        # ... add other features
    })

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data)[0]

    # Display the prediction
    if prediction == 1:
        st.success("The model predicts that the customer will respond to the campaign.")
    else:
        st.warning("The model predicts that the customer will not respond to the campaign.")


# You can add more elements to your Streamlit app, such as:
# - Explanations of the features
# - Visualizations of the model's performance
# - Information about the dataset
# - A section for uploading new data for prediction

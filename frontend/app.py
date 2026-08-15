#import necessary libraries
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for property features
Product_Weight = st.number_input("Product Weight (in decimal format)", min_value=1.0, max_value=100.0, value=12.5)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["No Sugar", "Low Sugar", "Regular"])
Product_Allocated_Area = st.number_input("Allocated Store Display Area (as percent of total store display area, in decimal format)", min_value=0.001, max_value=0.50, value=.06)
Product_Type = st.selectbox("Product Type", ["Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Household", "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks", "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"])
Product_MRP = st.number_input("Product Retail Price", min_value=1.00, max_value=500.00, value=150.00)
Store_Size = st.selectbox("Store Size", ["Small", "Medium","High"])
Store_Location_City_Type = st.selectbox("Store Location Standard of Living Tier", ["Tier 1", "Tier 2","Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type2", "Supermarket Type1","Departmental Store","Food Mart"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_Type': Product_Type,
    'Product_MRP': Product_MRP,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/sales", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Sales']
        st.success(f"Predicted Sales: {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/salesbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")

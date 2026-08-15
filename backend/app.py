# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_prediction_api = Flask("SuperKart Sales Predictor")

# Define a route for the home page (GET request)
@superkart_sales_prediction_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single property prediction (POST request)
@superkart_sales_prediction_api.post('/v1/sales')
def predict_sales():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing product-store details and returns
    the predicted sales as a JSON response.
    """
    # Get the JSON data from the request body
    product_store_data = request.get_json()

    # Convert numerical inputs to numeric values
    product_mrp = float(product_store_data["Product_MRP"])
    allocated_area = float(product_store_data["Product_Allocated_Area"])

    # Calculate engineered features
    mrp_x_allocated_area = product_mrp * allocated_area
    product_mrp_band = assign_mrp_band(product_mrp)

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': float(product_store_data["Product_Weight"]),
        'Product_Sugar_Content': product_store_data['Product_Sugar_Content'],
        'Product_Allocated_Area': allocated_area,
        'Product_Type': product_store_data['Product_Type'],
        'Product_MRP': product_mrp,
        'Store_Size': product_store_data['Store_Size'],
        'Store_Location_City_Type': product_store_data['Store_Location_City_Type'],
        'Store_Type': product_store_data['Store_Type'],
        'MRP_x_Allocated_Area': mrp_x_allocated_area,
        'Product_MRP_Band': product_mrp_band
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = saved_model.predict(input_data)[0]

    # Return the predicted sales
    return jsonify({'Predicted Sales': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_prediction_api.post('/v1/salesbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing product details for multiple stores
    and returns the predicted sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make prediction
    predicted_sales = saved_model.predict(input_data).tolist()

    #Create output to add row id column and include input data in csv for reference
    prediction_output = input_data.copy()
    prediction_output.insert(0,"Batch_Row_Id",np.arange(1, len(input_data) + 1))
    prediction_output["Predicted_Product_Store_Sales_Total"] = predicted_sales
    
    # Create a dictionary of predictions
    output_dict = dict(prediction_output)

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_prediction_api.run(debug=True)

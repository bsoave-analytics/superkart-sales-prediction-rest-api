# Import necessary libraries
from pathlib import Path
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_prediction_api = Flask("SuperKart Sales Predictor")

# Locate the model relative to this app.py file
BACKEND_DIRECTORY = Path(__file__).resolve().parent

MODEL_PATH = (
    BACKEND_DIRECTORY
    / "superkart_sales_prediction_model_v1_0.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model file was not found at: {MODEL_PATH}"
    )

# Load the fitted preprocessing and model pipeline once at startup
saved_model = joblib.load(MODEL_PATH)

#defining function for engineered features to calculate based on user input
MRP_BOUNDARIES = [
    -np.inf,
    126.00, #training Q1 boundary
    147.00, #training median boundary
    168.00, #training Q3 boundary
    np.inf
]

MRP_LABELS = [
    "Low",
    "Medium",
    "High",
    "Premium"
]


def assign_mrp_band(product_mrp):
    """Assign an MRP band using the training-data boundaries."""

    mrp_band = pd.cut(
        pd.Series([product_mrp]),
        bins=MRP_BOUNDARIES,
        labels=MRP_LABELS,
        include_lowest=True
    ).iloc[0]

    return str(mrp_band)

REQUIRED_RAW_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]


def add_engineered_features(data):
    """
    Validate the raw input data and create the engineered features
    expected by the saved model pipeline.
    """

    model_data = data.copy()

    missing_columns = [
        column
        for column in REQUIRED_RAW_COLUMNS
        if column not in model_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Uploaded CSV is missing required columns: "
            f"{missing_columns}"
        )

    # Ensure the source columns are numerical
    model_data["Product_MRP"] = pd.to_numeric(
        model_data["Product_MRP"],
        errors="raise"
    )

    model_data["Product_Allocated_Area"] = pd.to_numeric(
        model_data["Product_Allocated_Area"],
        errors="raise"
    )

    model_data["Product_Weight"] = pd.to_numeric(
        model_data["Product_Weight"],
        errors="raise"
    )

    # Engineer MRP and allocated-area interaction
    model_data["MRP_x_Allocated_Area"] = (
        model_data["Product_MRP"]
        * model_data["Product_Allocated_Area"]
    )

    # Apply the same MRP band function used for single predictions
    model_data["Product_MRP_Band"] = (
        model_data["Product_MRP"]
        .apply(assign_mrp_band)
    )

    return model_data

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

    # Convert NumPy float32 to a native Python float
    predicted_sales = float(predicted_sales)

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
    
    if "file" not in request.files:
        return jsonify({
            "error": "No CSV file was provided."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No CSV file was selected."
        }), 400

    try:
        # Read the uploaded CSV
        input_data = pd.read_csv(file)

        # Add the engineered features expected by the model
        model_input = add_engineered_features(input_data)

        # Generate predictions and convert to Python floats
        predicted_sales = (
            saved_model.predict(model_input)
            .astype(float)
            .tolist()
        )

        # Create output containing the original input fields
        prediction_output = input_data.copy()

        # Add a generated batch row identifier
        prediction_output.insert(
            0,
            "Batch_Row_Id",
            np.arange(1, len(input_data) + 1)
        )

        # Add the predictions
        prediction_output[
            "Predicted_Product_Store_Sales_Total"
        ] = predicted_sales

        # Convert each column to a JSON-compatible list
        output_dict = prediction_output.to_dict(
            orient="list"
        )

        return jsonify(output_dict)

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:
        superkart_sales_prediction_api.logger.exception(
            "Batch prediction failed"
        )

        return jsonify({
            "error": str(error)
        }), 500

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_prediction_api.run(debug=True)

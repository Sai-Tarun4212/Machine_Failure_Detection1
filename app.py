import base64
import pickle
import pandas as pd
import streamlit as st


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            
        }}
         
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)


add_bg_from_local('background.png')  # Replace with your image file name.


def predict_failure(input_data, file="machine_failure_model.pkl"):
    try:
        with open(file, 'rb') as f:
            preprocessor, model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: Model file '{file}' not found. Please train the model and save it as '{file}'.")
        return None

    input_df = pd.DataFrame([input_data])
    input_df['type'] = input_df['Product ID'].str[0]

    X_processed = preprocessor.transform(input_df.drop(['UDI', 'Product ID'], axis=1))
    prediction = model.predict(X_processed)[0]

    return prediction

st.title("Machine Failure Prediction")


st.subheader("Enter Machine Data")

udi = st.number_input("UDI (Unique ID)", min_value=1, step=1, help="Enter a unique identifier for the machine.")
product_id = st.text_input("Product ID (e.g., L123)", help="Enter the Product ID in the format 'Letter followed by numbers'.")

air_temp = st.number_input(
    "Air temperature [K]",
    help="Enter the air temperature in Kelvin. Typical range: 290-310 K."
)
process_temp = st.number_input(
    "Process temperature [K]",
    help="Enter the process temperature in Kelvin. Typical range: 300-320 K."
)
rotational_speed = st.number_input(
    "Rotational speed [rpm]",
    help="Enter the rotational speed in RPM. Typical range: 2500-3000 rpm."
)
torque = st.number_input(
    "Torque [Nm]",
    help="Enter the torque in Newton-meters. Typical range: 20-60 Nm."
)
tool_wear = st.number_input(
    "Tool wear [min]",
    help="Enter the tool wear in minutes. Typical range: 0-200 minutes."
)


if st.button("Predict"):
    input_data = {
        'UDI': udi,
        'Product ID': product_id,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
    }

    prediction = predict_failure(input_data)

    if prediction is not None:
        if prediction == 1:
            st.error("Predicted: Machine Failure")
        else:
            st.success("Predicted: No Machine Failure")

st.sidebar.header("Overview")
st.sidebar.markdown("""
This project centers around developing a machine learning-powered application to predict machine failures in a manufacturing setting. The core idea is to leverage sensor data and product information to anticipate potential breakdowns, enabling proactive maintenance and minimizing downtime.
""")
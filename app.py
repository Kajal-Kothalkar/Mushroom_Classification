import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Label Encoders
with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

# Streamlit UI
st.title("Mushroom Classification App 🍄")
st.write("Enter the mushroom features to check if it's **edible or poisonous**.")

# User input fields
input_features = {}
for column in label_encoders.keys():
    if column != "class":
        options = list(label_encoders[column].classes_)
        input_features[column] = st.selectbox(f"Select {column}:", options)

# Convert user input to numeric values
input_data = [label_encoders[col].transform([input_features[col]])[0] for col in input_features.keys()]
input_df = pd.DataFrame([input_data], columns=input_features.keys())

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Edible ✅" if prediction == 0 else "Poisonous ❌"
    st.success(f"The mushroom is: **{result}**")

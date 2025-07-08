import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load safe camp data
earthquake_safe_camps = pd.read_csv("earthquake_safe_camps.csv")
flood_safe_camps = pd.read_csv("flood_safe_camps.csv")

# Load training data for earthquake and flood prediction
# Assuming you have a dataset with features and target labels for prediction
earthquake_data = pd.read_csv("earthquake_prediction_data.csv")
flood_data = pd.read_csv("flood_prediction_data.csv")

# Extract features and targets from the datasets
X_earthquake = earthquake_data.drop(columns="label")  # features
y_earthquake = earthquake_data["label"]  # target for earthquake
X_flood = flood_data.drop(columns="label")  # features
y_flood = flood_data["label"]  # target for flood

# Standardize the features
scaler = StandardScaler()
X_earthquake_scaled = scaler.fit_transform(X_earthquake)
X_flood_scaled = scaler.fit_transform(X_flood)

# Train Logistic Regression models on the fly
earthquake_model = LogisticRegression()
earthquake_model.fit(X_earthquake_scaled, y_earthquake)

flood_model = LogisticRegression()
flood_model.fit(X_flood_scaled, y_flood)

# Page layout
st.title("Smart Disaster Management System")

# Sidebar for Prediction Input
st.sidebar.subheader("Disaster Prediction")
disaster_type = st.sidebar.selectbox("Select Disaster Type for Prediction:", ["Earthquake", "Flood"])

if disaster_type == "Earthquake":
    # Enter input features for earthquake prediction
    feature1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, step=0.1)
    feature2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, step=0.1)
    feature3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, step=0.1)
    
    if st.sidebar.button("Predict Earthquake Risk"):
        user_input = scaler.transform([[feature1, feature2, feature3]])
        prediction = earthquake_model.predict(user_input)
        st.sidebar.write(f"Earthquake Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")

elif disaster_type == "Flood":
    # Enter input features for flood prediction
    feature1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, step=0.1)
    feature2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, step=0.1)
    feature3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, step=0.1)

    if st.sidebar.button("Predict Flood Risk"):
        user_input = scaler.transform([[feature1, feature2, feature3]])
        prediction = flood_model.predict(user_input)
        st.sidebar.write(f"Flood Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")

# Map Plotting Section
st.subheader("Plot Safe Camps on Map")
latitude = st.number_input("Enter Latitude:", format="%.6f")
longitude = st.number_input("Enter Longitude:", format="%.6f")
camp_type = st.selectbox("Select Safe Camp Type:", ["Earthquake Safe Camp", "Flood Safe Camp"])
route_type = st.selectbox("Select Route Type:", ["Road Route", "Air Route (Helicopter)"])

# Function to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = (np.sin(dLat / 2) ** 2 + 
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
         np.sin(dLon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Choose the safe camp dataset based on the selected type
if camp_type == "Earthquake Safe Camp":
    safe_camps = earthquake_safe_camps
else:
    safe_camps = flood_safe_camps

# Find the nearest camp
safe_camps["distance"] = safe_camps.apply(lambda x: calculate_distance(latitude, longitude, x["latitude"], x["longitude"]), axis=1)
nearest_camp = safe_camps.loc[safe_camps["distance"].idxmin()]

# Display Map
m = folium.Map(location=[latitude, longitude], zoom_start=10)
folium.Marker([latitude, longitude], popup="Your Location", icon=folium.Icon(color="blue")).add_to(m)

# Add nearest camp marker
folium.Marker(
    [nearest_camp["latitude"], nearest_camp["longitude"]],
    popup=f"Nearest {camp_type} at {nearest_camp['latitude']}, {nearest_camp['longitude']}",
    icon=folium.Icon(color="red")
).add_to(m)

# Draw route
if route_type == "Road Route":
    folium.PolyLine([[latitude, longitude], [nearest_camp["latitude"], nearest_camp["longitude"]]], color="green").add_to(m)
else:
    folium.PolyLine([[latitude, longitude], [nearest_camp["latitude"], nearest_camp["longitude"]]], color="blue", dash_array="5, 5").add_to(m)

# Display Folium Map in Streamlit
st_folium(m, width=700, height=500)

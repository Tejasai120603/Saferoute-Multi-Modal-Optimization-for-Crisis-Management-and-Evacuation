import pandas as pd
import numpy as np
import joblib 

# Load pre-trained models and scaler
pipeline_earthquake = joblib.load("Earthquake_Model.pkl")
model_flood = joblib.load("Flood_Model.pkl")
# scaler_flood = joblib.load("Flood_Scaler.pkl")
# label_encoder_flood = joblib.load("Flood_LabelEncoder.pkl")  
# # Ensure you saved the label encoder

# Precaution Display Function for Earthquake
def display_precautions(magnitude, population_density, building_age):
    precautions = []
    if magnitude < 4:
        precautions.append("Minimal impact. No evacuation required.")
    elif 4 <= magnitude < 5:
        precautions.append("Minor impact. Remain alert.")
    elif 5 <= magnitude < 6:
        precautions.append("Moderate impact. Evacuate high-rise buildings in crowded areas.")
    elif 6 <= magnitude < 7:
        precautions.append("Significant impact. Evacuate densely populated high-rise buildings.")
    elif 7 <= magnitude < 8:
        precautions.append("Severe impact. Immediate evacuation of high-rise buildings.")
    else:
        precautions.append("Devastating impact. Large-scale evacuation needed.")
    if population_density == "high":
        precautions.append("Prepare for potential mass evacuations.")
    if building_age == "old (20-50 years)":
        precautions.append("Inspect older buildings for structural vulnerabilities.")
    return "\n".join(precautions)

# Main Prediction Function
def main():
    prediction_type = input("Enter 0 for Earthquake Prediction or 1 for Flood Prediction: ")
    
    if prediction_type == '0':
        # Earthquake Prediction
        input_data = {
            'depth': float(input("Enter depth (in km): ")),
            'magType': input("Enter magnitude type (e.g., mb, ml, etc.): "),
            'nst': int(input("Enter nst (number of seismic stations): ")),
            'gap': float(input("Enter gap (azimuthal gap in degrees): ")),
            'dmin': float(input("Enter dmin (minimum distance to the earthquake): ")),
            'rms': float(input("Enter RMS (root mean square of residuals): ")),
            'horizontalError': float(input("Enter horizontal error (in km): ")),
            'depthError': float(input("Enter depth error (in km): ")),
            'magError': float(input("Enter magnitude error: ")),
            'magNst': int(input("Enter magNst (number of reporting stations for magnitude): ")),
            'type': "earthquake"
        }
        
        input_df_eq = pd.DataFrame([input_data])
        for col in pipeline_earthquake.named_steps['preprocessor'].transformers[0][2]:
            if col not in input_df_eq.columns:
                input_df_eq[col] = np.nan
        
        prediction_eq = pipeline_earthquake.predict(input_df_eq)
        predicted_magnitude = 5.5 if prediction_eq[0] == 1 else 4.5
        precautions = display_precautions(predicted_magnitude, "medium", "old (20-50 years)")
        
        print(f"\nEarthquake Prediction: {'Magnitude > 5' if prediction_eq[0] == 1 else 'Magnitude ≤ 5'}")
        print("Precautions:\n", precautions)

    elif prediction_type == '1':
        # Flood Prediction
        user_input_flood = {
            'latitude': float(input("Enter latitude: ")),
            'longitude': float(input("Enter longitude: ")),
            'daily_rainfall': float(input("Enter daily rainfall: ")),
            'river_water_level': float(input("Enter river water level: ")),
            'soil_moisture_content': float(input("Enter soil moisture content: ")),
            'elevation': float(input("Enter elevation: ")),
            'slope': float(input("Enter slope: ")),
            'wind_speed': float(input("Enter wind speed (km/h): ")),
            'construction_activity_encoded': int(input("Enter construction activity encoded (0=Low, 1=Moderate, 2=High): ")),
            'population_density_x_encoded': int(input("Enter population density X encoded (0=Low, 1=Medium, 2=High): ")),
            'dams_reservoirs_encoded': int(input("Enter dams/reservoirs encoded (0=No, 1=Yes): ")),
            'drainage_system_encoded': int(input("Enter drainage system encoded (0=Poor, 1=Good): ")),
            'population_density_y_encoded': int(input("Enter population density Y encoded (0=Low, 1=Medium, 2=High): ")),
            'agricultural_activity_encoded': int(input("Enter agricultural activity encoded (0=No, 1=Yes): "))
        }
        
        input_df_fl = pd.DataFrame([user_input_flood])
        # input_df_fl_scaled = scaler_flood.transform(input_df_fl)
        # predicted_category_encoded = model_flood.predict(input_df_fl_scaled)
        # predicted_category = label_encoder_flood.inverse_transform(predicted_category_encoded)[0]
        
        # print(f"\nFlood Prediction: Water Level Category - {predicted_category}")
        print("Suggested precautions based on flood severity.")
    else:
        print("Invalid input! Enter 0 for Earthquake or 1 for Flood prediction.")

if __name__ == "__main__":
    main()

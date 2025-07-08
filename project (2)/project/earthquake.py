import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib 


# Load datasets
earthquake_data_path = 'filtered_earthquake_india.csv'  # Update with the actual path
flood_data_path = 'encoded_flood_data_with_waterlevel.csv'  # Update with the actual path
earthquake_data = pd.read_csv(earthquake_data_path)
flood_data = pd.read_csv(flood_data_path)

# Earthquake data preprocessing
earthquake_data_cleaned = earthquake_data.drop(columns=['time', 'place', 'latitude', 'longitude'])
earthquake_data_cleaned['target'] = (earthquake_data_cleaned['mag'] > 5).astype(int)
X_earthquake = earthquake_data_cleaned.drop(columns=['mag', 'target'])
y_earthquake = earthquake_data_cleaned['target']
categorical_cols_earthquake = [col for col in X_earthquake.select_dtypes(include=['object']).columns]
numerical_cols_earthquake = X_earthquake.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Earthquake pipelines
numeric_transformer_earthquake = SimpleImputer(strategy='mean')
categorical_transformer_earthquake = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])
preprocessor_earthquake = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_earthquake, numerical_cols_earthquake),
        ('cat', categorical_transformer_earthquake, categorical_cols_earthquake)
    ]
)
pipeline_earthquake = Pipeline(steps=[
    ('preprocessor', preprocessor_earthquake),
    ('model', LogisticRegression(max_iter=200))
])

# Train-test split and fit earthquake model
X_train_eq, X_test_eq, y_train_eq, y_test_eq = train_test_split(X_earthquake, y_earthquake, test_size=0.2, random_state=42)
pipeline_earthquake.fit(X_train_eq, y_train_eq)

# Flood data preprocessing
flood_data['waterlevel_category'] = pd.cut(
    flood_data['waterlevel'], bins=[-float("inf"), 10, 20, float("inf")], labels=['low', 'medium', 'high']
)
label_encoder_flood = LabelEncoder()
flood_data['waterlevel_category_encoded'] = label_encoder_flood.fit_transform(flood_data['waterlevel_category'])
X_flood = flood_data[['latitude', 'longitude', 'daily_rainfall', 'river_water_level', 'soil_moisture_content',
                      'elevation', 'slope', 'construction_activity_encoded', 'population_density_x_encoded',
                      'dams_reservoirs_encoded', 'drainage_system_encoded', 'population_density_y_encoded',
                      'wind_speed', 'agricultural_activity_encoded']]
y_flood = flood_data['waterlevel_category_encoded']
X_train_fl, X_test_fl, y_train_fl, y_test_fl = train_test_split(X_flood, y_flood, test_size=0.2, random_state=42)

# Flood model
scaler_flood = StandardScaler()
X_train_fl = scaler_flood.fit_transform(X_train_fl)
X_test_fl = scaler_flood.transform(X_test_fl)
model_flood = LogisticRegression(max_iter=1000)
model_flood.fit(X_train_fl, y_train_fl)

# Accuracy scores
earthquake_accuracy = accuracy_score(y_test_eq, pipeline_earthquake.predict(X_test_eq))
flood_accuracy = accuracy_score(y_test_fl, model_flood.predict(X_test_fl))
joblib.dump(model_flood, "Flood_Model.pkl")
joblib.dump(pipeline_earthquake, "Earthquake_Model.pkl")

# Display precautions based on earthquake magnitude
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

# Calculate flood risk score and water speed
def calculate_risk_score(row):
    score = 0
    if row['daily_rainfall'] > 50:
        score += 3
    if row['river_water_level'] > 5:
        score += 2
    if row['soil_moisture_content'] > 30:
        score += 1
    return score

def calculate_water_speed(slope, elevation):
    return (slope * 0.5 + elevation * 0.1)

# Main function to get user input and make predictions
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
            'type': input("Enter earthquake type (optional, e.g., 'shallow', 'intermediate', 'deep'): ")
        }
        
        input_df_eq = pd.DataFrame([input_data])
        for col in X_earthquake.columns:
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
        input_df_fl = input_df_fl[X_flood.columns]
        input_df_fl_scaled = scaler_flood.transform(input_df_fl)
        predicted_category_encoded = model_flood.predict(input_df_fl_scaled)
        predicted_category = label_encoder_flood.inverse_transform(predicted_category_encoded)[0]
        
        risk_score = calculate_risk_score(input_df_fl.iloc[0])
        alert_level = ""
        precautions = ""
        if risk_score >= 8:
            alert_level = "Critical Risk - Immediate action required."
            precautions = "Evacuate if necessary. Move to higher ground."
        elif risk_score >= 5:
            alert_level = "High Risk - Remain on alert."
            precautions = "Prepare for potential evacuation."
        else:
            alert_level = "Moderate Risk - Minimal action required."
            precautions = "Monitor updates regularly."

        water_speed = calculate_water_speed(input_df_fl.iloc[0]['slope'], input_df_fl.iloc[0]['elevation'])

        print(f"\nFlood Prediction: {predicted_category.capitalize()} Water Level")
        print(f"Alert Level: {alert_level}")
        print(f"Precautions: {precautions}")
        print(f"Estimated Water Speed: {water_speed:.2f} m/s")

    else:
        print("Invalid input. Please enter 0 or 1.")

# Execute main function
if __name__ == "__main__":
    main()

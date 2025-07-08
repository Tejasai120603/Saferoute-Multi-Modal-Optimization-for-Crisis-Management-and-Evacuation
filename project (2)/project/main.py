from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

earthquake_model = joblib.load(r"C:/Users/suman/Downloads/personal/projects/project (2) (1)/project (2)/project/Earthquake_Model.pkl")

flood_model = joblib.load("C:/Users/suman/Downloads/personal/projects/project (2) (1)/project (2)/project/Flood_Model.pkl")
flood_data_path = "C:/Users/suman/Downloads/personal/projects/project (2) (1)/project (2)/project/encoded_flood_data_with_waterlevel.csv"
flood_data = pd.read_csv(flood_data_path)

flood_data["waterlevel_category"] = pd.cut(
    flood_data["waterlevel"],
    bins=[-float("inf"), 10, 20, float("inf")],
    labels=["low", "medium", "high"],
)

label_encoder_flood = LabelEncoder()
flood_data["waterlevel_category_encoded"] = label_encoder_flood.fit_transform(
    flood_data["waterlevel_category"]
)
scaler_flood = StandardScaler()
X_flood = flood_data[
    [
        "latitude",
        "longitude",
        "daily_rainfall",
        "river_water_level",
        "soil_moisture_content",
        "elevation",
        "slope",
        "construction_activity_encoded",
        "population_density_x_encoded",
        "dams_reservoirs_encoded",
        "drainage_system_encoded",
        "population_density_y_encoded",
        "wind_speed",
        "agricultural_activity_encoded",
    ]
]
y_flood = flood_data["waterlevel_category_encoded"]
X_train_fl, X_test_fl, y_train_fl, y_test_fl = train_test_split(
    X_flood, y_flood, test_size=0.2, random_state=42
)
X_train_fl = scaler_flood.fit_transform(X_train_fl)


def calculate_risk_score(row):
    score = 0
    if float(row["daily_rainfall"]) > 50:
        score += 3
    if float(row["river_water_level"]) > 5:
        score += 2
    if float(row["soil_moisture_content"]) > 30:
        score += 1
    return score


def calculate_water_speed(slope, elevation):
    return float(slope) * 0.5 + float(elevation) * 0.1

def display_precautions(magnitude, population_density, building_age):
    precautions = []
    if magnitude < 4:
        precautions.append("Minimal impact. No evacuation required.")
    elif 4 <= magnitude < 5:
        precautions.append("Minor impact. Remain alert.")
    elif 5 <= magnitude < 6:
        precautions.append(
            "Moderate impact. Evacuate high-rise buildings in crowded areas."
        )
    elif 6 <= magnitude < 7:
        precautions.append(
            "Significant impact. Evacuate densely populated high-rise buildings."
        )
    elif 7 <= magnitude < 8:
        precautions.append(
            "Severe impact. Immediate evacuation of high-rise buildings."
        )
    else:
        precautions.append("Devastating impact. Large-scale evacuation needed.")
    if population_density == "high":
        precautions.append("Prepare for potential mass evacuations.")
    if building_age == "old (20-50 years)":
        precautions.append("Inspect older buildings for structural vulnerabilities.")
    return "\n".join(precautions)


@app.get("/")
async def main():
    return {"message": "Earthquake/Flood Prediction API"}


@app.post("/prediction")
async def predict(request: Request):
    body = await request.json()
    print(body)
    input_data = body["data"]

    if input_data["prediction_type"] == "earthquake":
        input_data["type"] = input_data["prediction_type"]
        input_df_eq = pd.DataFrame([input_data])
        input_df_eq = input_df_eq[
            [
                "depth",
                "magType",
                "nst",
                "gap",
                "dmin",
                "rms",
                "horizontalError",
                "depthError",
                "magError",
                "magNst",
                "type",
            ]
        ]
        for col in earthquake_model.named_steps["preprocessor"].transformers[0][2]:
            if col not in input_df_eq.columns:
                input_df_eq[col] = np.nan
        prediction_eq = earthquake_model.predict(input_df_eq)
        prediction = int(list(prediction_eq)[0])
        predicted_magnitude = 5.5 if prediction_eq[0] == 1 else 4.5
        precautions = display_precautions(
            predicted_magnitude, "medium", "old (20-50 years)"
        )

        return {
            "predicted_magnitude": predicted_magnitude,
            "precautions": precautions,
            "prediction": prediction,
        }

    elif input_data["prediction_type"] == "flood":
        input_df_fl = pd.DataFrame([input_data])
        input_df_fl = input_df_fl[
            [
                "latitude",
                "longitude",
                "daily_rainfall",
                "river_water_level",
                "soil_moisture_content",
                "elevation",
                "slope",
                "construction_activity_encoded",
                "population_density_x_encoded",
                "dams_reservoirs_encoded",
                "drainage_system_encoded",
                "population_density_y_encoded",
                "wind_speed",
                "agricultural_activity_encoded",
            ]
        ]
        input_df_fl_scaled = scaler_flood.transform(input_df_fl)
        predicted_category_encoded = flood_model.predict(input_df_fl_scaled)
        predicted_category = label_encoder_flood.inverse_transform(
            predicted_category_encoded
        )[0]

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

        water_speed = calculate_water_speed(
            input_df_fl.iloc[0]["slope"], input_df_fl.iloc[0]["elevation"]
        )

        print(f"\nFlood Prediction: {predicted_category.capitalize()} Water Level")
        print(f"Alert Level: {alert_level}")
        print(f"Precautions: {precautions}")
        print(f"Estimated Water Speed: {water_speed:.2f} m/s")

        return {
            "category": f"{predicted_category.capitalize()} Water Level",
            "alert_level": alert_level,
            "precautions": precautions,
            "water_speed": f"{round(water_speed, 2)} m/s",
        }
    else:
        raise HTTPException(status_code=500, detail="Invalid type of prediction")


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8000)
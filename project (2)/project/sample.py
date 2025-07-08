import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

# Earthquake pipeline
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
joblib.dump(pipeline_earthquake, "Earthquake_Model.pkl")

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
model_flood = LogisticRegression(max_iter=1000)
model_flood.fit(X_train_fl, y_train_fl)
joblib.dump(model_flood, "Flood_Model.pkl")

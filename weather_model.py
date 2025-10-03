# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# %%
EGYPT_CITIES_GAZETTEER = {
    "Cairo": {"lat": 30.0444, "lon": 31.2357},
    "Alexandria": {"lat": 31.2001, "lon": 29.9187},
    "Giza": {"lat": 29.9870, "lon": 31.2118},
    "Shubra El Kheima": {"lat": 30.1286, "lon": 31.2422},
    "Port Said": {"lat": 31.2653, "lon": 32.3019},
    "Suez": {"lat": 29.9737, "lon": 32.5263},
    "Luxor": {"lat": 25.6872, "lon": 32.6396},
    "Aswan": {"lat": 24.0889, "lon": 32.8998},
    "Mansoura": {"lat": 31.0364, "lon": 31.3807},
    "Tanta": {"lat": 30.7885, "lon": 31.0019},
    "Fayoum": {"lat": 29.3099, "lon": 30.8418},
    "Zagazig": {"lat": 30.5878, "lon": 31.5020},
    "Ismailia": {"lat": 30.6043, "lon": 32.2723},
    "Kafr El Sheikh": {"lat": 31.1154, "lon": 30.9419},
    "Damietta": {"lat": 31.4165, "lon": 31.8133},
    "Damanhur": {"lat": 31.0366, "lon": 30.4699},
    "Minya": {"lat": 28.1099, "lon": 30.7503},
    "Sohag": {"lat": 26.5569, "lon": 31.6948},
    "Qena": {"lat": 26.1613, "lon": 32.7265},
    "Hurghada": {"lat": 27.2579, "lon": 33.8116},
    "Sharm El Sheikh": {"lat": 27.9158, "lon": 34.3295},
    "Marsa Matruh": {"lat": 31.3529, "lon": 27.2372},
    "Arish": {"lat": 31.1316, "lon": 33.7984},
    "Benha": {"lat": 30.4593, "lon": 31.1834},
    "Beni Suef": {"lat": 29.0744, "lon": 31.0978},
    "Siwa": {"lat": 29.2041, "lon": 25.5195},
    "Dahab": {"lat": 28.5093, "lon": 34.5131},
    "Ras Gharib": {"lat": 28.3598, "lon": 33.0881},
    "Quesna": {"lat": 30.5289, "lon": 31.1444},
    "Faiyum": {"lat": 29.3099, "lon": 30.8418},
    "Edfu": {"lat": 24.9786, "lon": 32.8753},
    "Kom Ombo": {"lat": 24.4752, "lon": 32.9300},
    "New Cairo": {"lat": 30.0289, "lon": 31.4969}
}


# %%
def create_custom_weather_labels(df):
    """Applies a set of custom rules to create a detailed 'weather_condition' column."""
    print("Creating the 'weather_condition' target column using custom rules...")
    
    def weather_label(row):
        temp_max = row['temperature_2m_max']
        temp_min = row['temperature_2m_min']
        humidity = row['relative_humidity_2m_mean']
        sunshine = row['sunshine_duration']
        precip = row['precipitation_sum']
        wind = row['windspeed_10m_max']

        if precip > 1.0 or (precip > 0.1 and humidity > 80): return "Rainy"
        if wind >= 40: return "Very Windy"
        if wind >= 35: return "Windy"
        if temp_max >= 38: return "Extremely Hot"
        if temp_max >= 35: return "Very Hot"
        if temp_max >= 30: return "Hot"
        if temp_min <= 18: return "Cold"
        if temp_min <= 18: return "Very Cold"
        if sunshine > 30000 and precip == 0 and temp_max >= 30: return "Sunny & Hot"
        if sunshine > 30000 and precip == 0 and wind >= 30: return "Sunny & Windy"
        if temp_max >= 30 and humidity >= 70: return "Hot & Humid"
        if sunshine < 20000: return "Cloudy"
        return "Clear"

    df['weather_condition'] = df.apply(weather_label, axis=1)
    
    print("\nDistribution of Weather Conditions:")
    print(df['weather_condition'].value_counts(normalize=True) * 100)
    
    return df

# %%
def preprocess_data(input_file="egypt_cities_weather_full_3.csv"):
    """Reads data, cleans, and engineers seasonal features based on historical averages."""
    print("\n--- Starting Full Data Preprocessing ---")
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: '{input_file}'")
        return None, None

    df = pd.read_csv(input_file)
    available_cities = df['city'].unique().tolist()
    
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    df = create_custom_weather_labels(df)

    print("\nEngineering seasonal features (historical monthly and daily averages)...")
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['year'] = df.index.year

    features_to_average = ['temperature_2m_max', 'precipitation_sum', 'windspeed_10m_max', 'relative_humidity_2m_mean', 'sunshine_duration']
    monthly_avg = df.groupby(['city', 'month'])[features_to_average].transform('mean')
    df[[f'{col}_monthly_avg' for col in features_to_average]] = monthly_avg
    daily_avg = df.groupby(['city', 'day_of_year'])[features_to_average].transform('mean')
    df[[f'{col}_daily_avg' for col in features_to_average]] = daily_avg

    df = pd.get_dummies(df, columns=['city'], prefix='city')
    df.dropna(inplace=True)
    print("Preprocessing complete.")
    return df, available_cities

# %%
def train_model(df):
    """Trains a LightGBM classification model and returns it."""
    print("\n--- Starting Model Training ---")
    
    y = df['weather_condition']
    features_to_drop = [
        'weather_condition', 'temperature_2m_max', 'temperature_2m_min',
        'temperature_2m_mean', 'apparent_temperature_max', 'apparent_temperature_min',
        'apparent_temperature_mean', 'precipitation_sum', 'rain_sum', 'snowfall_sum',
        'windspeed_10m_max', 'windspeed_10m_mean', 'windgusts_10m_max',
        'relative_humidity_2m_max', 'relative_humidity_2m_min', 'relative_humidity_2m_mean',
        'shortwave_radiation_sum', 'sunshine_duration', 'et0_fao_evapotranspiration',
        'latitude', 'longitude'
    ]
    X = df.drop(columns=features_to_drop)
    model_columns = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    lgbm = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
    lgbm.fit(X_train, y_train)
    
    y_pred = lgbm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    
    print("--- Model Training Complete ---")
    return lgbm, model_columns


# %%
def make_prediction(prediction_date_str, prediction_place, historical_df, model, model_columns):
    """Reconstructs seasonal features from a date and place to make a prediction."""
    print(f"\n--- Making Prediction for {prediction_place} on {prediction_date_str} ---")
    
    prediction_date = pd.to_datetime(prediction_date_str)
    
    city_df = historical_df[historical_df['city'] == prediction_place].copy()
    if city_df.empty:
        print(f"ERROR: No historical data found for city '{prediction_place}'")
        return None

    features = {}
    features['day_of_year'] = prediction_date.dayofyear
    features['month'] = prediction_date.month
    features['day_of_week'] = prediction_date.dayofweek
    features['year'] = prediction_date.year

    features_to_average = ['temperature_2m_max', 'precipitation_sum', 'windspeed_10m_max', 'relative_humidity_2m_mean', 'sunshine_duration']
    
    monthly_data = city_df[city_df.index.month == prediction_date.month]
    for col in features_to_average:
        features[f'{col}_monthly_avg'] = monthly_data[col].mean()

    daily_data = city_df[city_df.index.dayofyear == prediction_date.dayofyear]
    for col in features_to_average:
        features[f'{col}_daily_avg'] = daily_data[col].mean()

    for col in model_columns:
        if 'city_' in col:
            features[col] = 1 if col == f'city_{prediction_place}' else 0
            
    prediction_df = pd.DataFrame([features])
    prediction_df = prediction_df[model_columns]

    if prediction_df.isnull().sum().sum() > 0:
        print("Warning: Could not generate all historical averages. Filling with 0.")
        prediction_df.fillna(0, inplace=True)

    prediction = model.predict(prediction_df)
    return prediction[0]

# %%
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth."""
    R = 6371  # Radius of Earth in kilometers
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat / 2) * sin(dLat / 2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) * sin(dLon / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# %%
def find_closest_city(lat, lon, known_cities_coords):
    """Find the nearest city from our known data."""
    closest_city = None
    min_distance = float('inf')
    for city, coords in known_cities_coords.items():
        distance = haversine(lat, lon, coords['lat'], coords['lon'])
        if distance < min_distance:
            min_distance = distance
            closest_city = city
    return closest_city


# %%
def initialize_model(input_file="egypt_cities_weather_full_3.csv"):
    """Initialize and train the model, return all necessary data."""
    print("Loading and preprocessing data...")
    
    # Load raw data for predictions
    raw_df_for_prediction = pd.read_csv(input_file)
    raw_df_for_prediction['time'] = pd.to_datetime(raw_df_for_prediction['time'])
    raw_df_for_prediction.set_index('time', inplace=True)

    # Process and train
    processed_df, available_cities = preprocess_data(input_file)

    if processed_df is None:
        raise Exception("Failed to preprocess data")
    
    print("Training model...")
    trained_model, columns_for_model = train_model(processed_df)
    
    # Build coordinate lookup
    known_cities_coords = {}
    city_coords_df = raw_df_for_prediction[['city', 'latitude', 'longitude']].drop_duplicates('city').set_index('city')
    for city, row in city_coords_df.iterrows():
        known_cities_coords[city] = {'lat': row['latitude'], 'lon': row['longitude']}
    
    print("Model initialized successfully!")
    
    return {
        'model': trained_model,
        'columns': columns_for_model,
        'raw_df': raw_df_for_prediction,
        'available_cities': available_cities,
        'known_cities_coords': known_cities_coords
    }

# %%
import pickle

# Train model as usual
model_data = initialize_model("egypt_cities_weather_full_3.csv")

# Save the trained model and metadata
with open("weather_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model saved to weather_model.pkl")




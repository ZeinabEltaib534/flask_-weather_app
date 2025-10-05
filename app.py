from flask import Flask, render_template, request, jsonify
import pickle
from datetime import datetime

# Import helper functions from weather_model (but not re-train)
from weather_model import (
    make_prediction, 
    EGYPT_CITIES_GAZETTEER,
    find_closest_city
)

app = Flask(__name__)

# Load pickled model
with open("weather_model.pkl", "rb") as f:
    model_data = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', cities=sorted(list(EGYPT_CITIES_GAZETTEER.keys())))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_place_input = data.get('city', '').strip()
        user_date = data.get('date', '').strip()
        
        # Validate date
        try:
            datetime.strptime(user_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD'}), 400

        prediction_place = user_place_input
        prediction_note = ""

        # Find closest city if not in dataset
        if prediction_place not in model_data['available_cities']:
            if prediction_place in EGYPT_CITIES_GAZETTEER:
                user_city_coords = EGYPT_CITIES_GAZETTEER[prediction_place]
                closest_city = find_closest_city(
                    user_city_coords['lat'], 
                    user_city_coords['lon'], 
                    model_data['known_cities_coords']
                )
                
                if closest_city:
                    prediction_note = f"Prediction based on nearest available city: {closest_city}"
                    prediction_place = closest_city
                else:
                    return jsonify({'error': f'Could not find a close city for {prediction_place}'}), 400
            else:
                return jsonify({'error': f'City {prediction_place} is not recognized'}), 400
        
        predicted_weather = make_prediction(
            user_date, 
            prediction_place, 
            model_data['raw_df'], 
            model_data['model'], 
            model_data['columns']
        )
        
        if predicted_weather:
            return jsonify({
                'success': True,
                'city': user_place_input,
                'date': user_date,
                'prediction': predicted_weather.title(),
                'note': prediction_note
            })
        else:
            return jsonify({'error': 'Failed to generate prediction'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)




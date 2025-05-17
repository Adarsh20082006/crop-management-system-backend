import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


df = pd.read_csv('Crop_recommendation.csv', index_col=0)
df.reset_index(inplace=True)


X = df[['N', 'P', 'K']]
y_temp = df['temperature']
temp_model = RandomForestRegressor()
temp_model.fit(X, y_temp)

def recommend_crop_and_conditions(n, p, k,count=3):
    filtered = df[
        (df['N'].between(n-5, n+5)) &
        (df['P'].between(p-5, p+5)) &
        (df['K'].between(k-5, k+5))
    ]
    
    if not filtered.empty:
        crops = filtered['label'].value_counts().head(count).index.tolist()
        temp = float(filtered['temperature'].mean())
        humidity = float(filtered['temperature'].mean())
        ph = float(filtered['ph'].mean())
        rainfall = float(filtered['rainfall'].mean())
    else:
        crops = ["Unknown"]
        temp = temp_model.predict([[n, p, k]])[0]
        humidity = ph = rainfall = None

    return {
        "recommended_crops": crops,
        "temperature": round(temp, 2),
        "humidity": round(humidity, 2) if humidity else "N/A",
        "ph": round(ph, 2) if ph else "N/A",
        "rainfall": round(rainfall, 2) if rainfall else "N/A"
    }

@app.route('/recommend-crop', methods=['POST'])
def recommend():
    data = request.get_json()
    n = float(data.get('N'))
    p = float(data.get('P'))
    k = float(data.get('K'))
    count = int(data.get('count', 3))

    result = recommend_crop_and_conditions(n, p, k,count)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
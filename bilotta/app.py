from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
    'key', 'time_signature', 'macro_genre', 'mode', 'year'
    ]



@app.route('/infer', methods=['POST'])
def ciao():
    data = request.get_json()
    input_data = data["input_data"]

    with open("model.pkl", "rb") as f:
        mymodel = pickle.load(f)

    try:
        input_data = [input_data[col] for col in feature_cols]
    except KeyError as e:
        return -1
    
    df = pd.DataFrame([input_data], columns=feature_cols)
    prediction = mymodel.predict(df)[0]

    response_data = {
        "result": {
            "value": prediction
        }
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  


try:
    with open('email_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    model, vectorizer = None, None  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model or vectorizer not loaded'})

      
        data = request.get_json()
        email_content = data.get('email_content', '').strip()

        if not email_content:
            return jsonify({'error': 'Email content is missing'})

 
        print(f"Received email content: {email_content}")

       
        email_vec = vectorizer.transform([email_content])

  
        prediction = model.predict(email_vec)

   
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()

    
        label_mapping = {0: 'Spam', 1: 'Personal', 2: 'Work', 3: 'Promotions'}
        predicted_label = label_mapping.get(prediction[0], 'Unknown')

        print(f"Prediction result: {predicted_label}")
        return jsonify({'prediction': predicted_label})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

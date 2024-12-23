from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
# Load the pre-trained model
filename = 'Finalized_CKD_Model.sav'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        albumin = float(request.form['albumin'])
        blood_glucose_random = float(request.form['blood_glucose_random'])
        blood_urea = float(request.form['blood_urea'])
        serum_creatine = float(request.form['serum_creatine'])
        hormone_level = float(request.form['hormone_level'])
        packed_cell_volume = float(request.form['packed_cell_volume'])
        white_blood_count = float(request.form['white_blood_count'])

        # Format the input data into a numpy array
        input_data = np.array([[albumin, blood_glucose_random, blood_urea, serum_creatine, hormone_level, packed_cell_volume, white_blood_count]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        print(prediction)
        # Map prediction to result
        result = 'Chronic Kidney Disease Detected' if prediction[0] == 1 else 'No Chronic Kidney Disease'
        #result=prediction
        print(result)

        return render_template('output.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


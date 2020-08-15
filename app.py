from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict' , methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 0)

    return render_template('index.html',
                            prediction_text = "The Wine quality is {}" .format(output))

if __name__=="__main__":
    app.run(debug=True)

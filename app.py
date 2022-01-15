import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle

app = flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_feature = [int(x) in request.form.values()]
    
    final_feature = [np.array(int_feature)]
    prediction = model.predict(final_feature)
    output = round(prediction[0],2)
    return render_template('index.html',prediction_text='time={}'.format(output))

if __name__ == '__main__':
    app.run(debug = True)

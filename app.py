from flask import Flask,render_template,request
import pickle
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)
mul_reg = open("heart.pkl", "rb")
model = joblib.load(mul_reg)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template("home.html",prediction=output)
if __name__ == '__main__':
    app.run(debug=True)

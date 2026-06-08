from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application

## import ridge regressor and scaler model
ridge_model = pickle.load(open('models/ridge_model.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/') ## index page
def index():
    return render_template('index.html')


@app.route('/home') ## home page
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST']) ## predict page
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain= float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes= float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        ## now after i have got the data i will stardardize it and then pass it to the model for prediction
        ## for that i have standard scalar model
        new_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        new_data_scaled = standard_scaler.transform(new_data)

        ## now after scaling the data i will pass it to the model for prediction
        prediction = ridge_model.predict(new_data_scaled)

        ## after that i will show the results in the same home page with the prediction result
        return render_template('home.html', prediction=prediction[0])


        

        
    else:
        return render_template('home.html')
    
    return render_template('predict.html')






if __name__ == '__main__':
    app.run(host="0.0.0.0")
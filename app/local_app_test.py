import requests
from sklearn.externals import joblib

"""
Guide to run this flask app test locally...

First, setup the app to run locally on development Flask server:
    - from terminal within app directory
    - % export FLASK_APP=main.py
    - % flask run
    
Next, send traffic to the app
    - run this script
    - loads training data
    - takes an instance of the training data, converts to json, sends it to the app
    - app recieves the POST request, and returns the score in JSON format

200 {"predict_prob": "0.00972129142988537", "predict_date": "2020-07-08 02:52:15.296447", "predict_flag": "0"}

"""

# grab the test data to test the API
df_test = joblib.load('data/pickles/gbm_model_x_test.sav')

# put data into a json object to mirror incoming data from API
df_test['json'] = df_test.apply(lambda x: x.to_json(), axis=1)

# point to local api host for testing
api_url = 'http://127.0.0.1:5000/'

# test send traffic to local server
data = df_test['json'][0]

# post request to the local api
r = requests.post(url=api_url, data=data)

# show what we get back
print(r.status_code, r.text)

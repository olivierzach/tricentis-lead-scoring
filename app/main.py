from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize


# initialize the flask app
app = Flask(__name__)

# load model assets
model = joblib.load('data/pickles/gbm_model.sav')
scale_fit = joblib.load('data/pickles/gbm_scaler.sav')
classification_threshold = joblib.load('data/pickles/gbm_classification_threshold.sav')


# app needs to know which code to run for each url request
# we map urls to python functions using the @route decorator
@app.route('/')
def index():
    return render_template('index.html')


# method to format data can return model score
@app.route('/', methods=['POST', 'GET'])
def get_score():
    if request.method == 'POST':

        # get the un-normalized data from the post request
        data = request.get_json(force=True)
        df = json_normalize(data)
        df.set_index('email', inplace=True)

        # scale the data with the scale learning from training
        df = scale_fit.fit_transform(df)

        # get probability prediction from the model
        prediction = model.predict_proba(df)[:, 1]

        # append prediction back to the data frame
        df['predict_prob'] = prediction

        # flag with custom classifier threshold
        df['predict_flag'] = np.where(
            df.predict_prob >= classification_threshold,
            1,
            0
        )

        # tag prediction with date
        df['predict_date'] = pd.to_datetime('now')

        # prepare output to give back
        return_cols = ['predict_prob', 'predict_date', 'predict_flag']
        res = df[return_cols].reset_index().to_json(orient='index')

        return res


if __name__ == '__main__':
    app.run()

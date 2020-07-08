from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json

# initialize the flask app
app = Flask(__name__)


# app needs to know which code to run for each url request
# we map urls to python functions using the @route decorator
@app.route('/')
def index():
    return render_template('index.html')


# method to format data can return model score
@app.route('/', methods=['POST', 'GET'])
def get_score():
    if request.method == 'POST':

        # load model assets
        model = joblib.load('model_assets/gbm_model.sav')
        scale_fit = joblib.load('model_assets/gbm_scaler.sav')
        classification_threshold = joblib.load('model_assets/gbm_classification_threshold.sav')

        # get the un-normalized data from the post request
        data = request.get_json(force=True)
        df = json_normalize(data)

        # scale the data with the scale learning from training
        df = scale_fit.fit_transform(df)

        # get probability prediction from the model
        prediction = model.predict_proba(df)[:, 1]

        # flag with custom classifier threshold
        predict_flag = np.where(
            prediction >= classification_threshold,
            1,
            0
        )

        # tag prediction with date
        predict_date = pd.to_datetime('now')

        # prepare output to give back as json
        return_data = {
            'predict_prob': str(prediction[0]),
            'predict_date': str(predict_date),
            'predict_flag': str(predict_flag[0])
        }

        return json.dumps(return_data)


if __name__ == '__main__':
    app.run()

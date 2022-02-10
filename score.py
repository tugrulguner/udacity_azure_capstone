import json
import numpy as np
import sklearn
import joblib

def init():
    
    global model
    model = joblib.load('./outputs/model/automl_best_model.pkl')

def run(input_data):
    
    data = np.array(json.loads(input_data)['data'], dtype=np.float32)
    
    # make prediction
    out = model.predict(data)

    return out
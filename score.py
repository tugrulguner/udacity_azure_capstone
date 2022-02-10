import json
import numpy as np
import os

def init():
    
    global model
    model = json.load('./outputs/model/automl_best_model.pkl')

def run(input_data):
    
    data = np.array(json.loads(input_data)['data'], dtype=np.float32)
    
    # make prediction
    out = model.predict(data)

    return out.tolist()
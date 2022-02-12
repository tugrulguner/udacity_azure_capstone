import json
import numpy as np
import sklearn
import joblib
import os
import pandas as pd

def init():
    
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model.pkl')
    model = joblib.load(model_path)

def run(input_data):
    
    data = pd.read_json(input_data, orient='records')
    # make prediction

    out = model.predict(data)
    return json.dumps({'Prediction': out.tolist()})

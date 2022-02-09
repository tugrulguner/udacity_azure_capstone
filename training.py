from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core.workspace import Workspace

def main():

    ws = Workspace.from_config()

    key = "Mobile Phone Data"

    dataset = ws.datasets[key] 

    ds = dataset.to_pandas_dataframe()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="n_estimators")
    parser.add_argument('--criterion', type=str, default='gini', help="criterion")
    parser.add_argument('--max_depth', type=int, default=1, help="max depth")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("criterion:", np.str(args.criterion))
    run.log("max_depth:", np.int(args.max_depth))

    y = ds.pop['price_range']

    x_train, x_test, y_train, y_test = train_test_split(ds,y,test_size=0.1)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=args.max_depth
        ).fit(x_train, y_train)

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/hyper_model.pkl')
    
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
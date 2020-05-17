import os
import pickle

import pandas as pd
from sklearn.externals import joblib
from nextbike.io.utils import get_data_path


def read_df(path=os.path.join(get_data_path(), 'input/<My_data>.csv'), **kwargs):
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except FileNotFoundError:
        print('Data file not found. Path was ' + path)


def read_model():
    path = os.path.join(get_data_path(), 'output/models.pkl')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def read_scaler():
    path = os.path.join(get_data_path(), 'output/scaler.save')
    scaler = joblib.load(path)
    return scaler

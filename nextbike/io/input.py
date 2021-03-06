import os
import pickle

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nextbike.io.utils import get_data_path


def read_df(path: str = os.path.join(get_data_path(), 'input/<My_data>.csv'), **kwargs) -> pd.DataFrame:
    """
    Method importing a DataFrame from a specified path
    :param path: A str pointing to the respective csv file
    :param kwargs: Additional kwargs for pandas' read_csv method
    :return: None
    """
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except FileNotFoundError:
        print('Data file not found. Path was ' + path)


def read_model(type: str = 'regressor'):
    """
    Method for reading in pickled models
    :param type: A string representing if type of model is related to duration, false booking or direction prediction
    :return: model instance
    """
    if type == 'regressor':
        path = os.path.join(get_data_path(), 'output/duration.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
    elif type == 'booking_filter':
        path = os.path.join(get_data_path(), 'output/booking_filter.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
    elif type == 'classifier':
        path = os.path.join(get_data_path(), 'output/direction.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
    return model


def read_encoder(type: str = 'label') -> LabelEncoder or OneHotEncoder:
    """
    Mehthod to read the classes of an encoder object for later use
    :return: Encoder Object containing the correct classes
    """
    path = os.path.join(get_data_path(), 'output')
    if type == 'label':
        encoder = joblib.load(os.path.join(path, 'classes.joblib'))
    elif type == 'season':
        encoder = joblib.load(os.path.join(path, 'season.joblib'))
    elif type == 'station':
        encoder = joblib.load(os.path.join(path, 'station.joblib'))

    return encoder

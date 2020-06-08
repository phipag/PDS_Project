import os
import pickle
from pathlib import Path
import pandas as pd
from nextbike.io import get_data_path
import joblib
from sklearn.preprocessing import LabelEncoder


def save_model(model, type: str = 'regressor') -> None:
    """
    Method for saving trained models to disc.
    :param model: A trained model instance
    :param type: A string representing if type of model is related to duration, false booking or destination prediction
    :return: None
    """
    if type == 'regressor':
        pickle.dump(model, open(os.path.join(get_data_path(), 'output/duration.pkl'), 'wb'))
    elif type == 'booking_filter':
        pickle.dump(model, open(os.path.join(get_data_path(), 'output/booking_filter.pkl'), 'wb'))
    elif type == 'classifier':
        pickle.dump(model, open(os.path.join(get_data_path(), 'output/destination.pkl'), 'wb'))


def create_dir_if_not_exists(path: str) -> None:
    """
    Method that creates the specified directory if it not already exists.
    :param path: Path pointing to the directory that should exist or be created.
    :return: None
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_predictions(predicted_data: pd.DataFrame, type: str = 'regressor') -> None:
    """
    Method that saves DataFrames containing the raw data as well as predictions
    :param predicted_data: A DataFrame containing raw data and predictions
    :param type: A string representing if type of model is related to duration, false booking or destination prediction
    :return: None
    """
    path = os.path.join(get_data_path(), 'output')
    create_dir_if_not_exists(path)
    if type == 'regressor':
        predicted_data.to_csv(os.path.join(path, 'duration_predictions.csv'), index=False)
    elif type == 'classifier':
        predicted_data.to_csv(os.path.join(path, 'destination_predictions.csv'), index=False)


def save_encoder(encoder: LabelEncoder, type: str = 'label') -> None:
    """
    Mehthod to save the classes of an encoder object for later use
    :param encoder: The encoder object that was fit and used to transform target features in classification
    :return: None
    """
    path = os.path.join(get_data_path(), 'output')
    if type == 'label':
        joblib.dump(encoder, os.path.join(path, 'classes.joblib'))
    elif type == 'season':
        joblib.dump(encoder, os.path.join(path, 'season.joblib'))
    elif type == 'station':
        joblib.dump(encoder, os.path.join(path, 'station.joblib'))


def combine_predictions() -> None:
    """
    Method combining duration and destination prediction and saving the file to disc.
    :return: None
    """
    # Trying to load data from duration prediction
    try:
        duration_predictions = pd.read_csv(os.path.join(get_data_path(), 'output/duration_predictions.csv'))
    except Exception as e:
        print('Data from duration prediction could not be loaded.')
        raise e

    # Trying to load data from destination prediction
    try:
        destination_predictions = pd.read_csv(os.path.join(get_data_path(), 'output/destination_predictions.csv'))
    except Exception as e:
        print('Data from destination prediction could not be loaded.')
        raise e

    # Concatenating both DataFrames and save them to disk
    final_df = pd.concat([duration_predictions, destination_predictions['destination']], axis=1)
    final_df.to_csv(os.path.join(get_data_path(), 'output/final_predictions.csv'))
    print('Prediction data was combined and saved to disc.')

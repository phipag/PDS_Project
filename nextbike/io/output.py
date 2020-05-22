import os
import pickle
from pathlib import Path
from nextbike.io.utils import get_data_path
import pandas as pd
from nextbike.io import get_data_path


def save_model(model, type='regressor'):
    if type == 'regressor':
        pickle.dump(model, open(os.path.join(get_data_path(), 'output/duration.pkl'), 'wb'))
    elif type == 'booking_filter':
        pickle.dump(model, open(os.path.join(get_data_path(), 'output/booking_filter.pkl'), 'wb'))
    elif type == 'classifier':
        pickle.dump(model, open(os.path.join(get_data_path(), 'output/destination.pkl'), 'wb'))


def create_dir_if_not_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_predictions(predicted_data, type='regressor'):
    path = os.path.join(get_data_path(), 'output')
    create_dir_if_not_exists(path)
    if type == 'regressor':
        predicted_data.to_csv(os.path.join(path, 'duration_predictions.csv'), index=False)
    elif type == 'classifier':
        predicted_data.to_csv(os.path.join(path, 'destination_predictions.csv'), index=False)


def combine_predictions():
    duration_predictions = pd.read_csv(os.path.join(get_data_path(), 'output/duration_predictions.csv'))
    destination_predictions = pd.read_csv(os.path.join(get_data_path(), 'output/destination_predictions.csv'))

    final_df = pd.concat([duration_predictions, destination_predictions['destination']], axis=1)

    final_df.to_csv(os.path.join(get_data_path(), 'output/final_predictions.csv'))

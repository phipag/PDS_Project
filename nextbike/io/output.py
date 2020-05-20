import os
import pickle
from pathlib import Path
from nextbike.io.utils import get_data_path


def save_model(model, type='regressor'):
    if type == 'regressor':
        pickle.dump(model, open(os.path.join(get_data_path(), 'output/model.pkl'), 'wb'))
    elif type == 'booking_filter':
        pickle.dump(model, open(os.path.join(get_data_path(), 'output/booking_filter.pkl'), 'wb'))


def create_dir_if_not_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

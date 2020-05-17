import os
import pickle
from pathlib import Path
from sklearn.externals import joblib
from nextbike.io.utils import get_data_path


def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), 'output/models.pkl'), 'wb'))


def create_dir_if_not_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_scaler(scaler):
    joblib.dump(scaler, os.path.join(get_data_path(), 'output/scaler.save'))

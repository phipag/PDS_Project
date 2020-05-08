import os
import pickle
from pathlib import Path

from nextbike.io.utils import get_data_path


def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), 'output/model.pkl'), 'wb'))


def create_dir_if_not_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

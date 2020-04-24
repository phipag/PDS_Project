import os

from nextbike.io import (
    get_data_path,
    read_file
)


def load_df():
    return read_file(os.path.join(get_data_path(), 'input/mannheim.csv'))

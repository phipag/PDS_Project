import os

import geopandas as gdp

from nextbike.io import (
    get_data_path,
    read_file
)


class Preprocessor:
    def __init__(self):
        self.gdf = None

    def load_gdf(self):
        df = read_file(os.path.join(get_data_path(), 'input/mannheim.csv'), index_col=0, parse_dates=['datetime'])
        self.gdf = gdp.GeoDataFrame(df, crs='EPSG:4326', geometry=gdp.points_from_xy(df['p_lng'], df['p_lat']))
        return self.gdf

# TODO: Define the preprocessing workflow in a method to automate it.

import geopandas as gpd

from nextbike.preprocessing.AbstractValidator import AbstractValidator
from nextbike.preprocessing.Preprocessor import Preprocessor


class Transformer(AbstractValidator):
    __preprocessor: Preprocessor
    __gdf: gpd.GeoDataFrame = None

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        if self.__gdf is None:
            raise UserWarning('Data frame is not initialized.')
        return self.__gdf

    def __init__(self, preprocessor: Preprocessor):
        try:
            if preprocessor.validate():
                self.__preprocessor = preprocessor
        except ValueError:
            raise ValueError('Preprocessor validation failed. Please make sure that the Preprocessor was successful.')

    def transform(self, validate=False) -> None:
        # Split the original data into a start and end trips data frame
        start_gdf = self.__preprocessor.gdf[self.__preprocessor.gdf['trip'] == 'start']
        end_gdf = self.__preprocessor.gdf[self.__preprocessor.gdf['trip'] == 'end']
        # Initialize a new GeoDataFrame and calculate the target columns with native pandas
        self.__gdf = gpd.GeoDataFrame(crs='EPSG:4326')
        self.__gdf['bike_number'] = start_gdf['b_number']
        self.__gdf['start_time'] = start_gdf['datetime']
        self.__gdf['weekend'] = start_gdf['datetime'].dt.dayofweek // 5 == 1
        self.__gdf['start_position'] = start_gdf['geometry']
        self.__gdf['duration'] = end_gdf['datetime'] - start_gdf['datetime']
        self.__gdf['end_time'] = end_gdf['datetime']
        self.__gdf['end_position'] = end_gdf['geometry']
        if validate:
            self.validate()

    def validate(self) -> bool:
        if len(self.__gdf) != len(self.__preprocessor.gdf) / 2:
            raise ValueError('Length of transformed data frame is expected to be half of the preprocessed data frame.')
        return True

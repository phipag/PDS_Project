import os

import geopandas as gpd

from nextbike.io import get_data_path
from nextbike.preprocessing.AbstractValidator import AbstractValidator
from nextbike.preprocessing.Preprocessor import Preprocessor


class Transformer(AbstractValidator):
    """
    This class transforms the preprocessed GeoDataFrame to the target format defined in the exercise.
    """
    __preprocessor: Preprocessor
    __gdf: gpd.GeoDataFrame = None

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """
        A computed property checking if the GeoDataFrame is initialized and returning it if initialized
        :return: GeoDataFrame
        :raises: UserWarning
        """
        if self.__gdf is None:
            raise UserWarning('Data frame is not initialized.')
        return self.__gdf

    def __init__(self, preprocessor: Preprocessor):
        """
        Initializes the Transformer class. A valid Preprocessor instance is required.
        :param Preprocessor preprocessor: Valid Preprocessor instance
        :raises: ValueError
        """
        try:
            if preprocessor.validate():
                self.__preprocessor = preprocessor
        except ValueError:
            raise ValueError('Preprocessor validation failed. Please make sure that the Preprocessor was successful.')

    def transform(self, validate: bool = False) -> None:
        """
        Transform the preprocessed GeoDataFrame to the target format.
        :param bool validate: Indicates whether a validation post-hook should be run or not.
        :return: None
        """
        # Split the original data into a start and end trips data frame
        start_gdf = self.__preprocessor.gdf[self.__preprocessor.gdf['trip'] == 'start'].reset_index(drop=True)
        end_gdf = self.__preprocessor.gdf[self.__preprocessor.gdf['trip'] == 'end'].reset_index(drop=True)
        # Initialize a new GeoDataFrame and calculate the target columns with native pandas
        self.__gdf = gpd.GeoDataFrame(crs='EPSG:4326')
        self.__gdf['bike_number'] = start_gdf['b_number']
        self.__gdf['start_time'] = start_gdf['datetime']
        self.__gdf['weekend'] = start_gdf['datetime'].dt.dayofweek // 5 == 1
        self.__gdf['start_position'] = start_gdf['geometry']
        self.__gdf['start_position_name'] = start_gdf['p_name']
        self.__gdf['duration'] = (end_gdf['datetime'] - start_gdf['datetime']).dt.seconds
        self.__gdf['end_time'] = end_gdf['datetime']
        self.__gdf['end_position'] = end_gdf['geometry']
        self.__gdf['end_position_name'] = end_gdf['p_name']
        self.__gdf['is_station'] = (start_gdf['p_place_type'] == 12) & (end_gdf['p_place_type'] == 12)
        if validate:
            self.validate()

    def save(self) -> None:
        """
        Saves the transformed GeoDataFrame as csv-file to the disk.
        :return: None
        """
        self.__gdf.to_csv(os.path.join(get_data_path(), 'output/mannheim_transformed.csv'), index=False)

    def validate(self) -> bool:
        """
        Validates whether the GeoDataFrame matches the target format or not
        :return: bool
        :raises: ValueError
        """
        if self.__gdf is None:
            raise ValueError('Cannot validate data frame of None type. Please transform the data frame first.')

        if len(self.__gdf) != len(self.__preprocessor.gdf) / 2:
            raise ValueError('Length of transformed data frame is expected to be half of the preprocessed data frame.')
        return True

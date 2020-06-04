import os
import warnings

import geopandas as gpd
import numpy as np

from nextbike.io import (
    get_data_path,
    read_df
)
from nextbike.preprocessing.AbstractValidator import AbstractValidator

warnings.simplefilter(action='ignore', category=FutureWarning)


class Preprocessor(AbstractValidator):
    """
    This class handles the preprocessing of the NextBike data.
    """
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

    def load_gdf(self, path: str = None) -> None:
        """
        Reads the raw DataFrame, transforms it to a GeoDataFrame and initializes the __gdf property.
        :type path: object A path that points to the .csv file
        :return: None
        """
        if path:
            df = read_df(path, index_col=0, parse_dates=['datetime'])
        else:
            df = read_df(os.path.join(get_data_path(), 'input/mannheim.csv'), index_col=0, parse_dates=['datetime'])
        self.__gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=gpd.points_from_xy(df['p_lng'], df['p_lat']))

    def clean_gdf(self, validate: bool = False) -> None:
        """
        Cleans the GeoDataFrame so that it contains valid booking but still in the original format.
        :param bool validate: Indicates whether a validation post-hook should be run or not.
        :return: None
        """
        # Fill NaN values with 0 and drop double bookings
        self.__gdf.fillna(0, inplace=True)
        self.__gdf.drop_duplicates(subset=['b_number', 'datetime'], inplace=True)
        # Remove all trips of type 'first' and 'last'
        self.__gdf = self.__gdf[(self.__gdf['trip'] != 'first') & (self.__gdf['trip'] != 'last')]
        # Load the GeoJSON boundary of Mannheim
        mannheim_boundary_gdf = gpd.read_file(os.path.join(get_data_path(), 'input/mannheim_boundary.geojson'),
                                              crs='EPSG:4326')
        # Remove all trips which are not within Mannheim (using shapely is faster than geopandas' spatial join)
        self.__gdf = self.__gdf[self.__gdf.within(mannheim_boundary_gdf['geometry'][0])]
        # Remove trips without corresponding start or end booking
        self.__fix_bookings()
        if validate:
            self.validate()

    def validate(self) -> bool:
        """
        Validates whether the GeoDataFrame has a semantically and syntactically correct structure.
        :return: bool
        :raises: ValueError
        """
        if self.__gdf is None:
            raise ValueError('Cannot validate data frame of None type. Please load a data frame first.')

        trips = np.array(self.__gdf['trip'])
        b_numbers = np.array(self.__gdf['b_number'])
        for i in range(len(trips) - 1):
            if trips[i] == 'start' and trips[i + 1] == 'end' and b_numbers[i] != b_numbers[i + 1]:
                raise ValueError(
                    'Validation error at index {}: The first booking of a bike cannot start with trip type '
                    '\'end\'.'.format(i))
            if trips[i] == trips[i + 1]:
                raise ValueError('Validation error at index {}: Two consecutive rows should not have the same trip '
                                 'type.'.format(i))
        return True

    def __fix_bookings(self) -> None:
        """
        Applies an algorithm which restores a semantically correct structure in the original data set.
        :return: None
        """
        # Sort the data frame by b_number and datetime to have the bookings for each according to the timeline
        self.__gdf.sort_values(by=['b_number', 'datetime'], inplace=True)
        # Reset the index so that numpy indices and pandas indices are synchronized
        self.__gdf.reset_index(drop=True, inplace=True)
        # Use numpy to execute the code in the Cython space
        trips = np.array(self.__gdf['trip'])
        b_numbers = np.array(self.__gdf['b_number'])
        # Use a hash set for distinct O(1) insertion operations
        delete_indices = set()
        # Iterate until the second last index because the sliding window is constructed by the interval [i, i + 1]
        for i in range(len(trips) - 1):
            # Special case: The trips of one bike should not end with a trip of type 'start' and the booking of the next
            # bike should not start with a trip of type 'end'
            if trips[i] == 'start' and trips[i + 1] == 'end' and b_numbers[i] != b_numbers[i + 1]:
                delete_indices.add(i)
                delete_indices.add(i + 1)
            # Regular case: Remove double start or end trips (the first of both for start and the last of both for end)
            if trips[i] == trips[i + 1]:
                i_delete = i if trips[i] == 'start' else i + 1
                delete_indices.add(i_delete)
        # Call pandas' internal drop method once in the end to hand over the execution to Cython again
        self.__gdf.drop(delete_indices, inplace=True)

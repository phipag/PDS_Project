import os

import geopandas as gpd
import numpy as np

from nextbike.io import (
    get_data_path,
    read_file
)


class Preprocessor:
    def __init__(self):
        self.__gdf: gpd.GeoDataFrame | None = None

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        if self.__gdf is None:
            raise UserWarning('Data frame is not initialized.')
        return self.__gdf

    def load_gdf(self) -> None:
        df = read_file(os.path.join(get_data_path(), 'input/mannheim.csv'), index_col=0, parse_dates=['datetime'])
        self.__gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=gpd.points_from_xy(df['p_lng'], df['p_lat']))

    def clean_gdf(self) -> None:
        # Fill NaN values with 0 and drop double bookings
        self.__gdf.fillna(0, inplace=True)
        self.__gdf.drop_duplicates(subset=['b_number', 'datetime'], inplace=True)
        # Load the GeoJSON boundary of Mannheim
        mannheim_boundary_gdf = gpd.read_file(os.path.join(get_data_path(), 'input/mannheim_boundary.geojson'),
                                              crs='EPSG:4326')
        # Remove all trips which are not within Mannheim (using shapely is faster than geopandas' spatial join)
        self.__gdf = self.__gdf[self.__gdf.within(mannheim_boundary_gdf['geometry'][0])]
        # Remove all trips of type 'first' and 'last'
        self.__gdf = self.__gdf[(self.__gdf['trip'] != 'first') & (self.__gdf['trip'] != 'last')]
        # Remove trips without corresponding start or end booking
        self.__fix_bookings()
        # TODO: Add a validator functionality here which throws an error if there is a start end trip mismatch
        #  (useful for CLI)

    def __fix_bookings(self) -> None:
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

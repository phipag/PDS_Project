import multiprocessing as mp
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from nextbike.io import get_data_path
from nextbike.preprocessing.Preprocessor import Preprocessor


def execute_geo_filtering(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Load the GeoJSON boundary of Mannheim
    mannheim_boundary_gdf = gpd.read_file(os.path.join(get_data_path(), 'input/mannheim_boundary.geojson'),
                                          crs='EPSG:4326')
    # Remove all trips which are not within Mannheim (using native shapely is faster than geopandas' spatial join)
    return gdf[gdf.within(mannheim_boundary_gdf['geometry'][0])]


class ParallelPreprocessor(Preprocessor):
    def _geo_filter_mannheim_trips(self) -> None:
        n_cpus = mp.cpu_count()
        n_processes = n_cpus if n_cpus and n_cpus > 1 else 1
        # Do not use parallel execution if the computer has less than 8 cores because the multiprocess overhead is too
        # much relative to the performance improvement
        if n_processes < 8:
            return super()._geo_filter_mannheim_trips()

        split_gdf = np.array_split(self._gdf, n_processes)
        print('Start {} processes ...'.format(n_processes))
        with mp.Pool(processes=n_processes) as pool:
            self._gdf = gpd.GeoDataFrame(pd.concat(pool.map(execute_geo_filtering, split_gdf), ignore_index=True),
                                         crs=self._gdf.crs)

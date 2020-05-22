import numpy as np
import pandas as pd

from nextbike import io
from nextbike.preprocessing import Transformer
from sklearn import preprocessing


class PrepareForPrediction:

    def duration_preparation(self, transformer: Transformer, training: bool = True) -> dict:
        """

        :param transformer:
        :param training:
        :return:
        """
        raw_data = transformer.gdf

        if training:
            raw_data.loc[(raw_data['duration'] <= 180) | (
                    raw_data['start_position'] == raw_data['end_position']), 'false_booking'] = 1
            raw_data.fillna(0.0, inplace=True)
            false_bookings_series = raw_data['false_booking']

            col_to_drop = ['bike_number', 'start_position', 'end_time', 'end_position', 'end_position_name',
                           'false_booking']
            prediction_data = raw_data.drop(columns=col_to_drop)

            prediction_data = self.create_time_features(prediction_data, sin_cos_transform=True)
            prepared_data = self.create_dummy_features(prediction_data)

            prepared_data = prepared_data.merge(false_bookings_series, left_index=True, right_index=True)

            features = prepared_data.drop(columns=['duration'])
            target = prepared_data['duration'].values.reshape(-1, 1)
            print('Preparation was sccessful.')
            return {'raw_data': raw_data, 'prepared_data': prepared_data, 'features': features,
                    'target': target}

        else:
            col_to_drop = ['bike_number', 'start_position', 'end_time',
                           'end_position', 'end_position_name', 'duration']
            prediction_data = raw_data.drop(columns=col_to_drop)

            prediction_data = self.create_time_features(prediction_data, sin_cos_transform=True)
            prepared_data = self.create_dummy_features(prediction_data)

            features = prepared_data.copy()
            rfc = io.read_model(type='booking_filter')
            false_bookings = rfc.predict(features)
            features = np.concatenate((features, np.vstack(false_bookings)), axis=1)
            return {'raw_data': raw_data, 'prepared_data': prepared_data, 'features': features,
                    'target': None}

    def classification_preparation(self, transformer: Transformer, training: bool = True) -> dict:

        raw_data = transformer.gdf

        university_stations = ['DHBW Mannheim - Campus Coblitzallee',
                               'A5 - Universität West', 'L1 - Schloss',
                               'DHBW Mannheim - Campus Käfertalerstr.',
                               'Universität Schloss', 'Universität Mensa',
                               'Universitätsklinik Mannheim - CampusRad',
                               'Hochschule Mannheim']

        if training:
            raw_data.loc[(raw_data['start_position_name'].isin(university_stations)), 'destination'] = 'to_university'
            raw_data.loc[(raw_data['end_position_name'].isin(university_stations)), 'destination'] = 'from_university'
            raw_data.loc[(raw_data['duration'] <= 180) | (
                    raw_data['start_position'] == raw_data['end_position']), 'destination'] = 'false'
            raw_data.fillna('not_university', inplace=True)

            prediction_data = raw_data[['start_time', 'weekend', 'is_station', 'destination']].copy()

            prediction_data = self.create_time_features(prediction_data, sin_cos_transform=False, season_as_OHE=True)

            features = prediction_data.drop(columns=['destination', 'start_time'])

            y = prediction_data['destination']

            le = preprocessing.LabelEncoder().fit(y)

            y = le.transform(y)

            return {'raw_data': raw_data, 'prepared_data': prediction_data, 'features': features,
                    'target': y, 'encoder': le}
        else:
            prediction_data = raw_data[['start_time', 'weekend', 'is_station']].copy()

            prediction_data = self.create_time_features(prediction_data, sin_cos_transform=False, season_as_OHE=True)

            features = prediction_data.drop(columns=['start_time'])

            return {'raw_data': raw_data, 'prepared_data': prediction_data, 'features': features,
                    'target': None, 'encoder': None}

    def create_time_features(self, prediction_data: pd.DataFrame, sin_cos_transform: bool = True,
                             season_as_OHE: bool = False) -> pd.DataFrame:

        # Creating an individual column for hour of the day
        prediction_data['HOUR'] = prediction_data.start_time.dt.strftime('%-H').astype('int')

        # Creating an individual column for week of the year
        prediction_data['WEEK_OF_YEAR'] = prediction_data.start_time.dt.strftime('%W').astype('int')

        # Creating an individual column for day of the week
        prediction_data['DAY_OF_WEEK'] = prediction_data.start_time.dt.strftime('%w').astype('int')
        if season_as_OHE:
            season_list = [0, 1, 2, 3]
        else:
            season_list = ['WINTER', 'SPRING', 'SUMMER', 'FALL']
        seasons = []
        for month in prediction_data.start_time.dt.strftime('%m').astype('int'):
            if month in [1, 2, 12]:
                seasons.append(season_list[0])
            elif month in [3, 4, 5]:
                seasons.append(season_list[1])
            elif month in [6, 7, 8]:
                seasons.append(season_list[2])
            elif month in [9, 10, 11]:
                seasons.append(season_list[3])
        prediction_data['season'] = seasons

        if sin_cos_transform:
            # Applying sine,cosine transformation on column hour to retain the cyclical nature
            prediction_data['HOUR_SIN'] = np.sin(prediction_data.HOUR * (2. * np.pi / 24))
            prediction_data['HOUR_COS'] = np.cos(prediction_data.HOUR * (2. * np.pi / 24))

            # Applying sine,cosine transformation on column WEEK_OF_YEAR to retain the cyclical nature
            prediction_data['WEEK_OF_YEAR_SIN'] = np.sin(prediction_data.WEEK_OF_YEAR * (2. * np.pi / 52))
            prediction_data['WEEK_OF_YEAR_COS'] = np.cos(prediction_data.WEEK_OF_YEAR * (2. * np.pi / 52))

            # Applying sine,cosine transformation on column DAY_OF_WEEK to retain the cyclical nature
            prediction_data['DAY_OF_WEEK_SIN'] = np.sin(prediction_data.DAY_OF_WEEK * (2. * np.pi / 7))
            prediction_data['DAY_OF_WEEK_COS'] = np.cos(prediction_data.DAY_OF_WEEK * (2. * np.pi / 7))

            prediction_data.drop(columns=['WEEK_OF_YEAR', 'DAY_OF_WEEK', 'HOUR', 'start_time'], axis=1, inplace=True)

        return prediction_data

    def create_dummy_features(self, prediction_data: pd.DataFrame) -> pd.DataFrame:
        station_dummies = pd.get_dummies(
            prediction_data.loc[prediction_data['is_station'] == True, 'start_position_name'])

        seasonal_dummies = pd.get_dummies(prediction_data['season'])

        prediction_data.drop(
            columns=['season', 'start_position_name'], axis=1, inplace=True)

        prediction_data = pd.concat([prediction_data, seasonal_dummies, station_dummies], axis=1)
        prediction_data.fillna(0.0, inplace=True)

        return prediction_data

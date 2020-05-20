import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nextbike import io
from nextbike.preprocessing import Transformer


class PrepareForPrediction:

    def duration_preparation(self, transformer: Transformer, training: bool = True,
                             scaler: StandardScaler = None) -> dict:
        """
    
        :param scaler: 
        :param training: 
        :param transformer:
        :return:
        """
        raw_data = transformer.gdf

        if training:
            raw_data.loc[(raw_data['duration'] <= 180) | (
                    raw_data["start_position"] == raw_data["end_position"]), 'false_booking'] = 1
            raw_data.fillna(0.0, inplace=True)
            false_bookings_series = raw_data['false_booking']

            col_to_drop = ['bike_number', 'start_position', 'end_time', 'end_position', 'end_position_name',
                           'false_booking']
            prediction_data = raw_data.drop(columns=col_to_drop)

            Q1 = prediction_data['duration'].quantile(0.25)
            Q3 = prediction_data['duration'].quantile(0.75)
            IQR = Q3 - Q1

            mask = prediction_data['duration'].between((Q1 - 1.5 * IQR), (Q3 + 1.5 * IQR), inclusive=True)
            prediction_data = prediction_data.loc[mask]
        else:
            col_to_drop = ['bike_number', 'start_position', 'end_time', 'end_position', 'end_position_name']
            prediction_data = raw_data.drop(columns=col_to_drop)

        prediction_data['start_time'] = pd.to_datetime(prediction_data.start_time)

        # Creating an individual column for hour of the day
        prediction_data['HOUR'] = prediction_data.start_time.dt.strftime('%-H').astype('int')

        # Creating an individual column for week of the year
        prediction_data['WEEK_OF_YEAR'] = prediction_data.start_time.dt.strftime('%W').astype('int')

        # Creating an individual column for day of the week
        prediction_data['DAY_OF_WEEK'] = prediction_data.start_time.dt.strftime('%w').astype('int')

        seasons = []
        for month in prediction_data.start_time.dt.strftime('%m').astype('int'):
            if month in [1, 2, 12]:
                seasons.append('WINTER')
            elif month in [3, 4, 5]:
                seasons.append('SPRING')
            elif month in [6, 7, 8]:
                seasons.append('SUMMER')
            elif month in [9, 10, 11]:
                seasons.append('FALL')
        prediction_data['season'] = seasons

        # Applying sine,cosine transformation on column hour to retain the cyclical nature
        prediction_data['HOUR_SIN'] = np.sin(prediction_data.HOUR * (2. * np.pi / 24))
        prediction_data['HOUR_COS'] = np.cos(prediction_data.HOUR * (2. * np.pi / 24))

        # Applying sine,cosine transformation on column WEEK_OF_YEAR to retain the cyclical nature
        prediction_data['WEEK_OF_YEAR_SIN'] = np.sin(prediction_data.WEEK_OF_YEAR * (2. * np.pi / 52))
        prediction_data['WEEK_OF_YEAR_COS'] = np.cos(prediction_data.WEEK_OF_YEAR * (2. * np.pi / 52))

        # Applying sine,cosine transformation on column DAY_OF_WEEK to retain the cyclical nature
        prediction_data['DAY_OF_WEEK_SIN'] = np.sin(prediction_data.DAY_OF_WEEK * (2. * np.pi / 7))
        prediction_data['DAY_OF_WEEK_COS'] = np.cos(prediction_data.DAY_OF_WEEK * (2. * np.pi / 7))

        # hier muss kein drop first weil es ja noch floating starts gibt
        station_dummies = pd.get_dummies(
            prediction_data.loc[prediction_data['is_station'] == True, 'start_position_name'])

        seasonal_dummies = pd.get_dummies(prediction_data['season'], drop_first=True)

        prediction_data.drop(
            columns=['WEEK_OF_YEAR', 'DAY_OF_WEEK', 'HOUR', 'start_time', 'season', 'start_position_name'],
            axis=1, inplace=True)

        prepared_data = pd.concat([prediction_data, seasonal_dummies, station_dummies], axis=1)

        prepared_data.fillna(0.0, inplace=True)

        if training:
            prepared_data = prepared_data.merge(false_bookings_series, left_index=True, right_index=True)

        features = prepared_data.drop(columns=['duration'])
        target = prepared_data['duration'].values.reshape(-1, 1)

        if training:
            scaler = StandardScaler().fit(features)
            features = scaler.transform(features)
            io.save_scaler(scaler)
            print('Preparation was sccessful, scaler was saved on disk.')
            return {"raw_data": raw_data, "prepared_data": prepared_data, "features": features,
                    "target": target, "scaler": scaler}
        else:
            svc = io.read_model(type='booking_filter')
            false_bookings = svc.predict(features)
            features = np.concatenate((features, np.vstack(false_bookings)), axis=1)
            features = scaler.transform(features)
            return {"raw_data": raw_data, "prepared_data": prepared_data, "features": features,
                    "target": target, "scaler": scaler}

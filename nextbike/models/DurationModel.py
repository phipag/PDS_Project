from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import numpy as np

from nextbike import io
from nextbike.preprocessing import Preprocessor, Transformer
from nextbike.models.Model import Model
import os


class DurationModel(Model):
    """

    """

    def __init__(self) -> None:
        """

        """
        super().__init__()
        self.raw_data = None
        self.predicted_data = None
        self.prepared_data = None
        self.model = None
        self.scaler = None
        self.feature_vector = None
        self.target_vector = None

    def load_from_csv(self, csv: str, training: bool = True) -> None:
        """

        :param csv:
        :return:
        """
        p = Preprocessor()
        p.load_gdf()
        p.clean_gdf()
        t = Transformer(p)
        t.transform()
        self.load_from_transformer(t, training)

    def load_from_transformer(self, transformer: Transformer, training: bool = True) -> None:
        """

        :param transformer:
        :return:
        """
        if not transformer.validate:
            print("Please use a valid transformer.")
        else:
            print('Transformer is valid. Preparing data for the models now.')

        self.raw_data = transformer.gdf

        if training:
            col_to_drop = ['bike_number', 'start_position', 'end_time', 'end_position', 'end_position_name']
            prediction_data = self.raw_data.drop(columns=col_to_drop).loc[
                (self.raw_data['duration'] > 180) | (self.raw_data["start_position"] != self.raw_data["end_position"])]
            z = np.abs(stats.zscore(prediction_data['duration']))
            prediction_data = prediction_data[(z < 3)]
        else:
            col_to_drop = ['bike_number', 'start_position', 'end_time', 'end_position', 'end_position_name']
            prediction_data = self.raw_data.drop(columns=col_to_drop)

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

        self.prepared_data = pd.concat([prediction_data, seasonal_dummies, station_dummies], axis=1)

        self.prepared_data.fillna(0.0, inplace=True)

        self.feature_vector = self.prepared_data.drop(columns=['duration'])
        y = self.prepared_data['duration'].values.reshape(-1, 1)

        self.scaler = StandardScaler().fit(y)
        self.target_vector = self.scaler.transform(y)
        io.save_scaler(self.scaler)

    def train(self, n_jobs: int = -1, random_state: int = 123) -> None:
        """

        :param n_jobs:
        :param random_state:
        :return:
        """
        model = RandomForestRegressor(n_jobs=n_jobs, random_state=random_state)
        print('Linear models created')
        print('Training...')
        model.fit(self.feature_vector, self.target_vector)
        print('Training was successful.')
        self.model = model
        io.save_model(model)

    def save_predictions(self) -> None:
        path = os.path.join(io.get_data_path(), 'output')
        io.create_dir_if_not_exists(path)
        self.predicted_data.to_csv(os.path.join(path, 'predictions.csv'), index=False)

    def predict(self, csv: str) -> None:
        """

        :return:
        """
        if self.model == None:
            self.model = io.read_model()

        if self.scaler == None:
            self.scaler = io.read_scaler()

        self.load_from_csv(csv, training=False)

        predictions = self.scaler.inverse_transform(self.model.predict(self.feature_vector))

        self.predicted_data = self.raw_data.copy()

        self.predicted_data['Predictions'] = predictions

        self.save_predictions()

        return self.predicted_data

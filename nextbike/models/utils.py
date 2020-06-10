import numpy as np
import pandas as pd
from sklearn import preprocessing

from nextbike import io
from nextbike.preprocessing import Transformer


def duration_preparation(transformer: Transformer, training: bool = True) -> dict:
    """
    Static method that that performs feature engineering steps for duration prediction and training. The following
    features are engineered:
        - Periodic quantities (Hour, Week) that are transformed using the sin-cos transformation so that time
        steps have the right distance to each other
        - Season as one-hot-encoded binary variables
        - Starting stations as one-hot-encoded binary variables
        - False Booking as a binary variable indicating whether the booking was not valid
    :param transformer: Valid transformer instance containing the transformed GeoDataFrame.
    :param training: A boolean indicating whether preparation is conducted for training. If false booking duration and
                     false booking will not be returned
    :return: A dict containing raw and prepared data as DataFrame as well as the feature and target vector
    """
    raw_data = transformer.gdf

    if training:
        # Identify false bookings in relation to the information provided by the VRN and create dedicated series
        raw_data.loc[(raw_data['duration'] <= 180) & (
                raw_data['start_position'] == raw_data['end_position']), 'false_booking'] = 1
        raw_data.fillna(0.0, inplace=True)

        # Drop features that cannot be known in prediction scenario and can therefore not be used for training
        col_to_drop = ['bike_number', 'start_position', 'end_time', 'end_position', 'end_position_name']
        prediction_data = raw_data.drop(columns=col_to_drop)

        # Perform feature engineering using the dedicated methods
        prediction_data = create_time_features(prediction_data, sin_cos_transform=True)
        prepared_data = create_dummy_features(prediction_data)

        # Create target and feature vector
        features = prepared_data.drop(columns=['duration'])
        target = prepared_data['duration'].values.reshape(-1, 1)

        print('Preparation was successful.')

        return {'raw_data': raw_data, 'prepared_data': prepared_data, 'features': features,
                'target': target}

    else:
        # Drop columns that are not useful or cannot be known in a real prediction scenario
        col_to_drop = ['bike_number', 'start_position', 'end_time',
                       'end_position', 'end_position_name', 'duration']
        prediction_data = raw_data.drop(columns=col_to_drop)

        # Perform feature engineering using the dedicated methods
        prediction_data = create_time_features(prediction_data, sin_cos_transform=True)
        prepared_data = create_dummy_features(prediction_data, training)

        # Create feature vector and predict and concatenate false bookings using the previously trained classifier
        features = prepared_data.copy()
        rfc = io.read_model(type='booking_filter')
        false_bookings = rfc.predict(features)
        features = np.concatenate((features, np.vstack(false_bookings)), axis=1)

        print('Preparation was successful.')

        return {'raw_data': raw_data, 'prepared_data': prepared_data, 'features': features,
                'target': None}


def classification_preparation(transformer: Transformer, training: bool = True) -> dict:
    """
    Static method that that performs feature engineering steps for direction prediction and training. The following
    features are engineered:
        - Periodic quantities (Hour, Week, Season) that are transformed using the sin-cos transformation so that time
        steps have the right distance to each other
    :param transformer: Valid transformer instance containing the transformed GeoDataFrame.
    :param training: A boolean indicating whether preparation is conducted for training (Default). If false booking
                     duration and false booking will not be returned
    :return: A dict containing raw and prepared data as DataFrame as well as the feature and target vector and the
             label encoder
    """
    raw_data = transformer.gdf

    # A list containing all the university related stations in Mannheim
    university_stations = ['DHBW Mannheim - Campus Coblitzallee', 'A5 - Universität West', 'L1 - Schloss',
                           'DHBW Mannheim - Campus Käfertalerstr.', 'Universität Schloss', 'Universität Mensa',
                           'Universitätsklinik Mannheim - CampusRad', 'Hochschule Mannheim']

    if training:
        # Assign trip direction labels for all trips there are four classes in total:
        # ['to_university', 'from_unversity','false', 'not_university']
        raw_data.loc[(raw_data['start_position_name'].isin(university_stations)), 'direction'] = 'to_university'
        raw_data.loc[(raw_data['end_position_name'].isin(university_stations)), 'direction'] = 'from_university'
        raw_data.loc[(raw_data['duration'] <= 180) & (
                raw_data['start_position'] == raw_data['end_position']), 'direction'] = 'false'
        raw_data.fillna('not_university', inplace=True)

        # Keep only usable columns for training
        prediction_data = raw_data[['start_time', 'weekend', 'is_station', 'direction']].copy()

        # Perform feature engineering using the dedicated methods
        prediction_data = create_time_features(prediction_data, sin_cos_transform=False, season_as_ohe=False)

        # Create the feature vector
        features = prediction_data.drop(columns=['direction', 'start_time'])

        # Create the target vector and transform it to ordinal labels
        y = prediction_data['direction']
        le = preprocessing.LabelEncoder().fit(y)
        y = le.transform(y)

        # Save encoder to disk for later usage
        io.save_encoder(le)

        print('Preparation was successful.')

        return {'raw_data': raw_data, 'prepared_data': prediction_data, 'features': features,
                'target': y, 'encoder': le}
    else:
        # Keep only usable columns for prediction
        prediction_data = raw_data[['start_time', 'weekend', 'is_station']].copy()

        # Perform feature engineering using the dedicated methods
        prediction_data = create_time_features(prediction_data, sin_cos_transform=False, season_as_ohe=False)

        # Create the feature vector
        features = prediction_data.drop(columns=['start_time'])

        print('Preparation was successful.')

        return {'raw_data': raw_data, 'prepared_data': prediction_data, 'features': features,
                'target': None, 'encoder': None}


def create_time_features(prediction_data: pd.DataFrame, sin_cos_transform: bool = True,
                         season_as_ohe: bool = True) -> pd.DataFrame:
    """
    A dedicated method for time related feature engineering. Different time features are extracted
    (Hour, Week, Day, Season) of which the first three are transformed using the sin-cos transformation
    to map times on the unit-circle.
    :param prediction_data: A DataFrame containing the pre-prepared data
    :param sin_cos_transform: A boolean indicating whether time related features should be sin-cos transformed
    :param season_as_ohe: A boolean indicating whether the feature season should be one-hot-encoded
    :return: DataFrame containing the pre-prepared data as well as the engineered features
    """
    # Creating an individual column for hour of the day
    prediction_data['HOUR'] = prediction_data.start_time.dt.strftime('%-H').astype('int')

    # Creating an individual column for week of the year
    prediction_data['WEEK_OF_YEAR'] = prediction_data.start_time.dt.strftime('%W').astype('int')

    # Creating an individual column for day of the week
    prediction_data['DAY_OF_WEEK'] = prediction_data.start_time.dt.strftime('%w').astype('int')

    # Check whether Season is meant to be one-hot-encoded or used as ordinal variable
    if season_as_ohe:
        season_list = ['WINTER', 'SPRING', 'SUMMER', 'FALL']
    else:
        season_list = [0, 1, 2, 3]

    # Looping through the data assigning the right season in relation to the month
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

        # Dropping left-over time related columns
        prediction_data.drop(columns=['WEEK_OF_YEAR', 'DAY_OF_WEEK', 'HOUR', 'start_time'], axis=1, inplace=True)

    return prediction_data


def create_dummy_features(prediction_data: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    A dedicated method for the creation of one-hot-encoded binary features. If needed transforms real stations and
    season.
    :param prediction_data: A DataFrame containing the pre-prepared data
    :return: DataFrame containing the pre-prepared data as well as the engineered features
    """

    if training:
        # Create dummies of all real stations
        station_ohe = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
        station_dummies = station_ohe.fit_transform(
            prediction_data.loc[prediction_data['is_station'] == True, 'start_position_name'].to_numpy().reshape(-1, 1))
        station_dummies = pd.DataFrame(station_dummies, columns=station_ohe.get_feature_names())
        # Save station encoder to disk for later usage
        io.save_encoder(station_ohe, type='station')

        # Create dummies of the seasons
        seasonal_ohe = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
        seasonal_dummies = seasonal_ohe.fit_transform(prediction_data['season'].to_numpy().reshape(-1, 1))
        seasonal_dummies = pd.DataFrame(station_dummies, columns=seasonal_ohe.get_feature_names())
        # Save season encoder to disk for later usage
        io.save_encoder(seasonal_ohe, type='season')
    else:
        # Read station encoder from disk
        station_ohe = io.read_encoder(type='station')
        station_dummies = station_ohe.transform(
            prediction_data.loc[prediction_data['is_station'] == True, 'start_position_name'].to_numpy().reshape(-1, 1))
        station_dummies = pd.DataFrame(station_dummies, columns=station_ohe.get_feature_names())

        # Read season encoder from disk
        seasonal_ohe = io.read_encoder(type='season')
        seasonal_dummies = seasonal_ohe.transform(prediction_data['season'].to_numpy().reshape(-1, 1))
        seasonal_dummies = pd.DataFrame(station_dummies, columns=seasonal_ohe.get_feature_names())

    # Drop season and station names as no longer needed
    prediction_data.drop(columns=['season', 'start_position_name'], axis=1, inplace=True)

    # Concatenate the dummy variable vectors to the DataFrame and fill up empty cells which do not relate to a station
    prediction_data = pd.concat([prediction_data, seasonal_dummies, station_dummies], axis=1)
    prediction_data.fillna(0.0, inplace=True)

    return prediction_data

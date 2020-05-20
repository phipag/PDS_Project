from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from nextbike import io
from nextbike.preprocessing import Preprocessor, Transformer, Features
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
        self.features = None
        self.target = None
        self.predictions = None

    def load_from_csv(self, path: str = None, training: bool = True) -> None:
        """

        :param training:
        :param path:
        :return: None
        """
        p = Preprocessor()
        p.load_gdf(path=path)
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
            print("Transformation was not successful.")
        else:
            print('Transformation was successful. Conducting feature engineering now.')
            engineer = Features.PrepareForPrediction()
            contents = engineer.duration_preparation(transformer, training)

            self.raw_data = contents['raw_data']
            self.prepared_data = contents['prepared_data']
            self.features = contents['features']
            self.target = contents['target']

    def train(self, n_jobs: int = -1, random_state: int = 123, train_filter=True) -> None:
        """

        :param n_jobs:
        :param random_state:
        :return:
        """
        duration_model = RandomForestRegressor(n_jobs=n_jobs, random_state=random_state)
        print('RandomForest model is initialized with n_jobs: {} and random_state: {}'.format(n_jobs, random_state))
        print('Conducting training on {} rows'.format(len(self.target)))
        duration_model.fit(self.features, self.target.ravel())
        print('Training was successful.')
        self.model = duration_model
        io.save_model(duration_model)
        print('The model was saved on disk.')

        if train_filter:
            self.train_filter()

    def train_filter(self):
        rfc = RandomForestClassifier(n_jobs=-1)
        X = self.prepared_data.drop(columns=['false_booking', 'duration'])
        y = self.prepared_data['false_booking']
        rfc.fit(X, y)
        io.save_model(rfc, type='booking_filter')

    def save_predictions(self) -> None:
        path = os.path.join(io.get_data_path(), 'output')
        io.create_dir_if_not_exists(path)
        self.predicted_data.to_csv(os.path.join(path, 'predictions.csv'), index=False)

    def predict(self, csv: str, apply_filter=True) -> None:
        """

        :return:
        """
        if self.model is None:
            print('This DurationModel instance does not have a model loaded. Loading model from "data/output.')
            self.model = io.read_model()

        self.load_from_csv(csv, training=False)

        self.predictions = self.model.predict(self.features)

        self.predicted_data = self.raw_data.copy()

        self.predicted_data['Predictions'] = self.predictions

        self.save_predictions()

        return self.predicted_data

    def training_score(self):
        mae = mean_absolute_error(self.target.reshape(-1, 1), self.predictions)
        print('The mean absolute error is: {}'.format(mae))

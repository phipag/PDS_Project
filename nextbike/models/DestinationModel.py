from nextbike.models.Model import Model
from nextbike.preprocessing import Preprocessor, Transformer, Features
from sklearn.ensemble import RandomForestClassifier
from nextbike import io
import os
from sklearn.metrics import classification_report


class DestinationModel(Model):

    def __init__(self):
        super().__init__()
        self.raw_data = None
        self.predicted_data = None
        self.prepared_data = None
        self.model = None
        self.features = None
        self.target = None
        self.predictions = None
        self.encoder = None

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
        :param training:
        :return:
        """
        if not transformer.validate:
            print("Transformation was not successful.")
        else:
            print('Transformation was successful. Conducting feature engineering now.')
            engineer = Features.PrepareForPrediction()
            contents = engineer.classification_preparation(transformer, training)

            self.raw_data = contents['raw_data']
            self.prepared_data = contents['prepared_data']
            self.features = contents['features']
            self.target = contents['target']
            self.encoder = contents['encoder']

    def train(self, n_jobs: int = -1, random_state: int = 123) -> None:
        """

        :return:
        """
        destination_model = RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
        print('RandomForestClassifier model is initialized with n_jobs: {} and random_state: {}'.format(n_jobs,
                                                                                                        random_state))
        print('Conducting training on {} rows'.format(len(self.target)))
        destination_model.fit(self.features, self.target.ravel())
        print('Training was successful.')
        self.model = destination_model
        io.save_model(destination_model, type='classifier')
        print('The model was saved on disk.')

    def predict(self, path: str = None, save=False) -> None:
        """

        :param path:
        :param save:
        :return:
        """
        if self.model is None:
            print('This DurationModel instance does not have a model loaded. Loading model from "data/output.')
            self.model = io.read_model(type='classifier')
        if path is not None:
            self.load_from_csv(path, training=False)

        self.predictions = self.model.predict(self.features)

        self.predicted_data = self.raw_data.copy()

        self.predicted_data['destination'] = self.predictions

        if save:
            io.save_predictions(self.predicted_data, type='classifier')

        return self.predicted_data

    def training_score(self):
        target_transformed = self.encoder.inverse_transform(self.target)
        pred_transformed = self.encoder.inverse_transform(self.predictions)
        c_rep = classification_report(target_transformed, pred_transformed)
        print(c_rep)

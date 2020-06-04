from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error

from nextbike import io
from nextbike.models import utils
from nextbike.models.Model import Model
from nextbike.preprocessing import Preprocessor, Transformer


class DurationModel(Model):
    """
    Class for the training/prediction of booking duration models inheriting from the abstract class Model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.raw_data = None  # Data obtained from preliminary transformation step
        self.predicted_data = None  # Raw data with added predictions
        self.prepared_data = None  # Prepared data containing all necessary transformed/engineered features
        self.model = None  # Machine learning model
        self.features = None  # Vector containing the features
        self.target = None  # Vector containing the target
        self.predictions = None  # Vector containing the predictions

    def load_from_csv(self, path: str = None, training: bool = True) -> None:
        """
        Method that allows the user to load data from a .csv file from a specified path.
        After transformation steps this data is passed on to load_from_transformer to initiate preparation steps.
        :param path: A str pointing to the respective csv file
        :param training: Boolean indicating whether data should be prepared for training
        :return: None
        """
        # Conduct pre-processing steps
        p = Preprocessor()
        p.load_gdf(path=path)
        p.clean_gdf()

        # Conduct transformation steps
        t = Transformer(p)
        t.transform()

        # Pass transformed data to load from transformer for final preparation
        self.load_from_transformer(t, training)

    def load_from_transformer(self, transformer: Transformer, training: bool = True) -> None:
        """
        Method that allows loading data from a valid transformer instance. If the transformer is valid data is passed to
        preparation method to conduct data preparation for duration prediction. Afterwards processed data is assigned
        to the respective class attributes.
        :param transformer: Valid transformer instance containing the transformed GeoDataFrame.
        :param training: Boolean indicating whether data should be prepared for training
        :return: None
        """
        if not transformer.validate():
            print('Transformation was not successful.')
        else:
            print('Transformation was successful. Conducting feature engineering now.')
            contents = utils.duration_preparation(transformer, training)

            self.raw_data = contents['raw_data']
            self.prepared_data = contents['prepared_data']
            self.features = contents['features']
            self.target = contents['target']

    def train(self, n_jobs: int = -1, random_state: int = 123) -> None:
        """
        Method for duration prediction model training utilizing sklearn's RandomForestRegressor
        :param n_jobs: An int specifying the number of threads to be used for training. Default: All available threads
        :param random_state: An int to lock the randomness of the model for reproducibility. If None results can vary
        :return:
        """
        # Initialize the model with given parameters
        duration_model = RandomForestRegressor(n_jobs=n_jobs, random_state=random_state)
        print('RandomForest model is initialized with n_jobs: {} and random_state: {}'.format(n_jobs, random_state))

        # Initiating the training process
        print('Conducting training for duration prediction on {} rows'.format(len(self.target)))
        duration_model.fit(self.features, self.target.ravel())
        print('Training was successful.')

        # Assigning the trained model to the respective class attribute
        self.model = duration_model

        # Save the trained model to disc
        print('Saving the duration model to disk.')
        io.save_model(duration_model)

        # Train a classifier that allows for the use of false bookings in later prediction
        print('Initiating training process for false booking classification.')
        self.train_filter(n_jobs, random_state)

        print('Training process is complete.')

    def train_filter(self, n_jobs: int = -1, random_state: int = 123):
        """
        A method that trains a RandomForestClassifier so that false bookings can be predicted without using duration
        information that is not available in real-world prediction scenarios. This model will be used in the preparation
        method to predict and merge false booking information to the data for duration prediction.
        :return: None
        """
        # Initialize the model
        rfc = RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
        print(
            'RandomForestClassifier is initialized with n_jobs: {} and random_state: {}'.format(n_jobs, random_state))

        # Create the feature and target vector
        X = self.prepared_data.drop(columns=['false_booking', 'duration'])
        y = self.prepared_data['false_booking']

        # Initiating the training process
        print('Conducting training for false booking classification on {} rows'.format(len(X)))
        rfc.fit(X, y)
        print('Training was successful.')

        # Save the trained model to disk
        print('Saving the classification model to disk.')
        io.save_model(rfc, type='booking_filter')

    def predict(self, path: str = None, save=False) -> None:
        """
        A method for prediction.
        :param path: A str pointing to the respective csv file containing the data that should be predicted (optional).
                     Only needed if Instance was not used for training.
        :param save: A boolean whether predicted data should be saved to output directory
        :return: A DataFrame containing the raw data as well as predictions
        """
        # Load duration prediction model if class instance was not used for training
        if self.model is None:
            print('This DurationModel instance does not have a model loaded. Loading model from "data/output".')
            self.model = io.read_model()

        # Predict new data if class instance was not used for training
        if path is not None:
            print('A path was specified. Initiating transformation and preparation steps.')
            self.load_from_csv(path, training=False)
        elif self.features is None:
            raise BaseException('This instance does not contain any data. You have to load data by specifying a path.')
        else:
            print('Using data contained by the class instance.')

        # Conduct predictions
        print('Starting prediction process.')
        self.predictions = self.model.predict(self.features)
        print('Prediction was successful.')

        # Create a DataFrame containing predictions and the transformed data
        self.predicted_data = self.raw_data.copy()
        self.predicted_data['predictions'] = self.predictions

        # Saving the predictions to disk
        if save:
            print('Saving prediction data to disk.')
            io.save_predictions(self.predicted_data)

        print('The prediction process is completed.')
        return self.predicted_data

    def training_score(self):
        """
        Mehtod that can be used after training and prediction of the same instance to return the mean absoulte error of
        predictions made on the training set.
        :return: None
        """
        # Try to calculate the mean absolute error if class instance contains all relevant information
        try:
            mae = mean_absolute_error(self.target.reshape(-1, 1), self.predictions)
            print('The mean absolute error is: {}'.format(mae))
        except AttributeError:
            print('There is no target vector present. Note that this method can only '
                  'be used by instances that were used for training ')

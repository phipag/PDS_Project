from abc import ABC, abstractmethod
from nextbike.preprocessing import Transformer
from pandas import DataFrame


class Model(ABC):
    """
    Abstract class forcing child classes that are for model creation to implement standardized methods.
    """

    @abstractmethod
    def __init__(self):
        """
        Method that forces child classes to obtain the following attributes.
        """
        self.raw_data = None  # Data obtained from preliminary transformation step
        self.predicted_data = None  # Raw data with added predictions
        self.prepared_data = None  # Prepared data containing all necessary transformed/engineered features
        self.model = None  # Machine learning model
        self.features = None  # Vector containing the features
        self.target = None  # Vector containing the target
        self.predictions = None  # Vector containing the predictions

    @abstractmethod
    def load_from_csv(self, path: str = None, training: bool = True) -> None:
        """
        Method that allows the user to load data from a .csv file from a specified path. After transformation steps this
        data is passed on to load_from_transformer to initiate preparation steps.
        :param path: A str pointing to the respective csv file
        :param training: Boolean indicating whether data should be prepared for training
        :return: None
        """
        pass

    @abstractmethod
    def load_from_transformer(self, transformer: Transformer, training: bool = True) -> None:
        """
        Method that allows loading data from a valid transformer instance. If the transformer is valid data is passed to
        preparation method to conduct data preparation for prediction.
        :param transformer: Valid transformer instance containing the transformed GeoDataFrame.
        :param training: Boolean indicating whether data should be prepared for training
        :return: None
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Method that starts model training.
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, path: str) -> DataFrame:
        """
        Method that initiates prediction.
        :param path:  A str pointing to the respective csv file
        :return: A DataFrame containing the raw data as well as predictions
        """
        pass

    @abstractmethod
    def training_score(self):
        """
        Method that prints the training score.
        :return:
        """
        pass

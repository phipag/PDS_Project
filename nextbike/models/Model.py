from abc import ABC, abstractmethod
from nextbike.preprocessing import Transformer


class Model(ABC):
    """

    """

    @abstractmethod
    def __init__(self):
        self.raw_data = None
        self.predicted_data = None
        self.prepared_data = None
        self.model = None
        self.features = None
        self.target = None
        self.predictions = None

    @abstractmethod
    def load_from_csv(self, path: str = None, training: bool = True) -> None:
        """

        :param path:
        :param training:
        :return:
        """
        pass

    @abstractmethod
    def load_from_transformer(self, transformer: Transformer, training: bool = True) -> None:
        """

        :param transformer:
        :param training:
        :return:
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """

        :return:
        """
        pass

    @abstractmethod
    def save_predictions(self) -> None:
        """

        :return:
        """
        pass

    @abstractmethod
    def predict(self, path: str) -> None:
        """

        :param path:
        :return:
        """
        pass

    @abstractmethod
    def training_score(self):
        pass

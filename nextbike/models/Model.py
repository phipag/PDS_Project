from abc import ABC, abstractmethod


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

    @abstractmethod
    def load_from_csv(self, csv) -> None:
        """

        :param csv:
        :return:
        """
        pass

    @abstractmethod
    def load_from_transformer(self, transformer) -> None:
        """

        :param transformer:
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
    def predict(self, csv) -> None:
        """

        :param csv:
        :return:
        """
        pass

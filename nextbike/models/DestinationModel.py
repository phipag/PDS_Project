from nextbike.models.Model import Model
from nextbike.preprocessing import Transformer


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

    def load_from_csv(self, path: str = None, training: bool = True) -> None:
        """

        :param path:
        :param training:
        :return:
        """
        pass

    def load_from_transformer(self, transformer: Transformer, training: bool = True) -> None:
        """

        :param transformer:
        :param training:
        :return:
        """
        pass

    def train(self) -> None:
        """

        :return:
        """
        pass

    def save_predictions(self) -> None:
        """

        :return:
        """
        pass

    def predict(self, path: str) -> None:
        """

        :param path:
        :return:
        """
        pass

    def training_score(self):
        pass

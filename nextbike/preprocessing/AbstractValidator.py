from abc import (
    ABC,
    abstractmethod
)


class AbstractValidator(ABC):
    """
    This is an abstract class which forces the implementation of a validate method in child classes.
    """

    @abstractmethod
    def validate(self) -> bool:
        """
        This method should either return true if the validation (of what so ever) was successful or raise an Error.
        :return: bool
        """
        pass

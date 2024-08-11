from glayout.syntaxer.process_input import Session
from abc import ABC, abstractmethod

class Command(ABC):
    def __init__(self, session: Session):
        self.session = session
        pass

    @abstractmethod
    def manipulate_code(self):
        """This function is here to manipulate the Code object of a session
        It doesn't return anything, it just manipulates the code class
        """
        pass

    @abstractmethod
    def action(self):
        """this function is here to complete actions that don't affect the code class
        Doesn't return anything either
        """
        pass

    @abstractmethod
    def pattern_matcher(self) -> bool:
        """this function here is to return is the command matches a certain pattern
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """this is just a variable describing the name of the class
        """
        pass



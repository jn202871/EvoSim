from abc import ABC, abstractmethod

class Controller(ABC):
    @abstractmethod
    def act(self, ind: dict, grid: dict, pos: tuple) -> tuple:
        pass

class Logger(ABC):
    def on_step(self, step: int, sim) -> None:
        pass

    def on_extinction(self, step: int, sim) -> None:
        pass


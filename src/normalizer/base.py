from abc import ABC, abstractmethod


class TextNormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Recibe texto en español colombiano y devuelve
        texto en español neutro.
        """
        pass

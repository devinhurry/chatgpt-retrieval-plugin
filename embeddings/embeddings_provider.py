from abc import ABC, abstractmethod
from typing import List

class EmbeddingsProvider(ABC):

    @abstractmethod
    def get_embeddings(self, texts: List[str]) ->  List[List[float]]:
        """
        Takes in a list of texts and returns a list of embeddings.
        """
        raise NotImplementedError    


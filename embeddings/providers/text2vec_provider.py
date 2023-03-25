from typing import List
from embeddings.embeddings_provider import EmbeddingsProvider

class Text2VecProvider(EmbeddingsProvider):
    model_name = None
    model = None
    def __init__(self, model_name):
        self.model_name = model_name

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Takes in a list of texts and returns a list of embeddings.
        """
        from text2vec import SentenceModel

        if self.model is None:
            self.model = SentenceModel(self.model_name)
        embeddings = self.model.encode(texts)
        result = []
        for e in embeddings:
            result.append(e.tolist())
        return result
     
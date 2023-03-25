from typing import List
from embeddings.embeddings_provider import EmbeddingsProvider
import services.openai as openai


class OpenAIProvider(EmbeddingsProvider):
    def get_embeddings(self, texts: List[str]) -> List[ List[List[float]]]:
        """
        Takes in a list of texts and returns a list of embeddings.
        """
        return openai.get_embeddings(texts)

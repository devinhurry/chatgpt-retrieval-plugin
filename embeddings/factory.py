import os
from embeddings.embeddings_provider import EmbeddingsProvider

# q:如何初始化global变量
# a:https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
# global embeddings_provider

def get_embeddings_provider() -> EmbeddingsProvider:
    if get_embeddings_provider.provider is not None:
        return get_embeddings_provider.provider 
    embeddings_provider_name = os.environ.get("EMBEDDINGS_PROVIDER")
    assert embeddings_provider_name is not None

    match embeddings_provider_name:
        case "openai":
            from embeddings.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider()
            get_embeddings_provider.provider  = provider
            return provider
        case "text2vec":
            text2vec_model = os.environ.get("TEXT2VEC_MODEL")
            assert text2vec_model is not None
            
            from embeddings.providers.text2vec_provider import Text2VecProvider
            provider = Text2VecProvider(text2vec_model)
            get_embeddings_provider.provider  = provider
            return provider
        case _:
            raise ValueError(f"Unsupported embedding provider: {embeddings_provider_name}")
        
get_embeddings_provider.provider = None




from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into embeddings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Convert a single query into an embedding.
        """
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding
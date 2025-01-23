from sentence_transformers import SentenceTransformer 
import numpy as np
from typing import List

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        return self.model.encode(text).tolist()
        
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

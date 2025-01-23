import time 
import logging
import uuid
from typing import Dict, List
from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter
from sentence_transformers import SentenceTransformer
from ..models.topic_models import Topic, TopicAttribute
from datetime import datetime, timezone
import torch

class QdrantService:
    def __init__(self):
        # Initialize Qdrant client with configurable URL and API key
        self.client = self._initialize_client()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Load the sentence transformer model
        self._ensure_collection_exists()

    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client with configurable URL and API key."""
        qdrant_url = "https://d3015961-e04b-4057-8be9-21b2f28c9894.europe-west3-0.gcp.cloud.qdrant.io"
        qdrant_api_key = "tfyVEahRthudwnXUysoDmdGPh-2u-LLmTsdWzdiTsWRBJqyr7nqupA"
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("Qdrant URL or API Key is missing")
        
        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30  
        )

    def _ensure_collection_exists(self):
        """Ensure the 'topics' collection exists with correct vector dimensions and payload index."""
        try:
            collections = self.client.get_collections()
            if "topics" not in [t.name for t in collections.collections]:
                self.client.create_collection(
                    collection_name="topics",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # For MiniLM-L6-v2 embeddings
                )
                logging.info("Collection 'topics' created successfully.")
            else:
                logging.info("Collection 'topics' already exists.")
        except Exception as e:
            logging.error(f"Failed to ensure collection exists: {str(e)}", exc_info=True)
            raise

    def check_topic_exists_with_similarity(self, new_topic: Topic, similarity_threshold: float = 0.7) -> bool:
        """Check if a similar topic already exists in the database."""
        topic_embedding = self.model.encode(new_topic.topic).tolist()
        results = self.client.search(
            collection_name="topics",
            query_vector=topic_embedding,
            limit=5
        )
        
        for result in results:
            existing_topic_data = result.payload
            existing_topic = Topic(
                topic=existing_topic_data['topic'],
                attributes=TopicAttribute(**existing_topic_data['attributes'])
            )
            similarity_score = self.calculate_similarity(new_topic, existing_topic)
            if similarity_score >= similarity_threshold:
                logging.info(f"Similar topic found: {existing_topic.topic} with similarity {similarity_score}")
                return True
        return False

    def calculate_similarity(self, topic1: Topic, topic2: Topic) -> float:
        """Calculate similarity between two topics based on their attributes."""
        topic1_vector = self.model.encode(topic1.topic).tolist()
        topic2_vector = self.model.encode(topic2.topic).tolist()
        return self._cosine_similarity(topic1_vector, topic2_vector)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def store_topic(self, topic: Topic) -> Dict:
        """Store a topic in Qdrant database with similarity check."""
        if self.check_topic_exists_with_similarity(topic):
            return {
                "status": "error",
                "message": "A similar topic already exists in the database."
            }
        
        vector = self.model.encode(topic.topic).tolist()
        payload = {
            "topic": topic.topic,
            "attributes": {
                "field": topic.attributes.field,
                "sub_field": topic.attributes.sub_field,
                "subject_matter": topic.attributes.subject_matter,
                "relevance": topic.attributes.relevance,
                "potential_impact": topic.attributes.potential_impact,
                "hotness": topic.attributes.hotness
            },
            "stored_at": datetime.now(timezone.utc).isoformat()
        }

        self.client.upsert(
            collection_name="topics",
            points=[{
                "id": str(uuid.uuid4()),  # Unique ID for each topic
                "payload": payload,
                "vector": vector         
            }]
        )

        logging.info(f"Successfully stored topic: {topic.topic}")
        return {
            "status": "success",
            "message": "Topic stored successfully",
            "topic": payload
        }

    async def delete_topic(self, topic_name: str) -> Dict:
        """Delete a topic from Qdrant database by topic name."""
        try:
            delete_filter = Filter(must=[{"key": "topic", "match": {"value": topic_name}}])
            await self.client.delete(
                collection_name="topics",
                points_selector=delete_filter
            )
            logging.info(f"Successfully deleted topic: {topic_name}")
            return {
                "status": "success", 
                "message": f"Topic '{topic_name}' deleted successfully"
            }
        except Exception as e:
            logging.error(f"Error deleting topic {topic_name}: {str(e)}", exc_info=True)
            return {
                "status": "error", 
                "message": f"Failed to delete topic: {str(e)}"
            }

    async def query_topics(self, query: str) -> List[Topic]:
        """Query the database for topics relevant to the user's question."""
        query_vector = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name="topics",
            query_vector=query_vector,
            limit=5
        )
        
        topics = []
        for result in results:
            topic_data = result.payload
            topics.append(Topic(
                topic=topic_data['topic'],
                attributes=TopicAttribute(**topic_data['attributes'])
            ))
        
        return topics

    async def get_all_topics(self) -> List[Topic]:
        """Retrieve all topics from the database, sorted by most recently stored first."""
        try:
            topics = []
            next_page_offset = None

            while True:
                results, next_page_offset = self.client.scroll(
                    collection_name="topics",
                    limit=1000,
                    order_by={"key": "stored_at", "direction": "desc"},
                    offset=next_page_offset
                )

                if not results:
                    break

                for point in results:
                    payload = point.payload
                    topics.append(Topic(
                        topic=payload['topic'],
                        attributes=TopicAttribute(**payload['attributes'])
                    ))

                if not next_page_offset:
                    break

            return topics
        except Exception as e:
            logging.error(f"Error retrieving topics: {str(e)}", exc_info=True)
            return []
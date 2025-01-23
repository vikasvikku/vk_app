import json
from fastapi import HTTPException
from .openai_service import OpenAIService
from .qdrant_service import QdrantService
from .content_service import ContentService
from .embedding_service import EmbeddingService
from ..utils.text_chunker import TextChunker
from ..models.topic_models import TopicAttribute, TopicResponse, TextInput, InputType, Topic
from typing import List, Tuple, Dict
import logging

class TopicService:
    def __init__(self):
        self.openai_service = OpenAIService()
        self.content_service = ContentService()
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()

    async def process_input(self, input_data: TextInput) -> TopicResponse:
        """Process input data, extract topics, and optionally store them in Qdrant."""
        try:
            # Extract content based on input type
            text, tables, images = await self._extract_content(input_data)
            
            # Chunk text for more granular topic extraction
            chunks = TextChunker.chunk_text(text, 1000)
            
            # Extract topics from text chunks
            all_topics = []
            for chunk in chunks:
                topics_result = self.openai_service.extract_topics(chunk)
                all_topics.extend(topics_result.parsed.topics)
            
            # Extract topics from tables if present
            if tables:
                table_topics = self._process_tables(tables)
                all_topics.extend(table_topics)
            
            # Extract topics from images if present
            if images:
                image_topics = self._process_images(images)
                all_topics.extend(image_topics)
            
            # Remove duplicate topics
            unique_topics = self._remove_duplicates(all_topics)
            
            return TopicResponse(
                topics=unique_topics,
                message="Topics extracted successfully",
                status="success"
            )
        
        except Exception as e:
            logging.error(f"Error in process_input: {str(e)}", exc_info=True)
            return TopicResponse(
                topics=[],
                message=f"Error processing input: {str(e)}",
                status="error"
            )
    def _is_similar_topic(self, new_topic: Topic, existing_topics: List[Topic]) -> bool:
        """Check if the new topic is similar to any existing topics."""
        for existing_topic in existing_topics:
            if new_topic.topic == existing_topic.topic:
                return True
            # You can add more sophisticated similarity checks here if needed
        return False

    def _remove_duplicates(self, topics: List[Topic]) -> List[Topic]:
     """Remove duplicate topics based on topic name and attributes."""
     seen = set()
     unique_topics = []
    
     for topic in topics:
        # Create a unique identifier tuple that includes topic name and its attributes
        identifier = (
            topic.topic,
            topic.attributes.field,
            topic.attributes.sub_field,
            topic.attributes.subject_matter,
            topic.attributes.relevance,
            topic.attributes.potential_impact,
            topic.attributes.hotness
        )
        
        if identifier not in seen:
            seen.add(identifier)
            unique_topics.append(topic)

     return unique_topics
        

    async def _extract_content(self, input_data: TextInput) -> Tuple[str, List[str], List[str]]:
        """Extract text, tables, and images based on input type."""
        if input_data.input_type == InputType.TEXT:
            return input_data.content, [], []
        elif input_data.input_type == InputType.URL:
            text, tables = await self.content_service.extract_text_from_url(input_data.content)
            return text, tables, []
        elif input_data.input_type == InputType.PDF:
            return await self.content_service.extract_text_from_pdf(input_data.content)
        else:
            raise ValueError(f"Unsupported input type: {input_data.input_type}")

    def _process_tables(self, tables: List[str]) -> List[Topic]:
        """Process table data to extract relevant concepts."""
        topics = []
        for table in tables:
            table_topics_result = self.openai_service.extract_topics(table)
            topics.extend(table_topics_result.parsed.topics)
        return topics

    def _process_images(self, images: List[str]) -> List[Topic]:
        """Process image data to extract relevant topics."""
        topics = []
        for image in images:
            extracted_text =  self.content_service.extract_text_from_pdf(image)
            image_topics_result =  self.openai_service.extract_topics(extracted_text)
            topics.extend(image_topics_result.parsed.topics)
        return topics

    async def store_selected_topics(self, selected_topics: List[Topic]) -> Dict:
        """Store selected topics with similarity check."""
        storage_results = []
        
        for topic in selected_topics:
            result = self.qdrant_service.store_topic(topic)
            storage_results.append(result)
        
        successful_topics = [
            result['topic'] for result in storage_results 
            if result['status'] == 'success'
        ]
        failed_topics = [
            result['message'] for result in storage_results 
            if result['status'] == 'error'
        ]
        
        return {
            "status": "success" if successful_topics else "similar_topics_found",
            "message": "Topics processed",
            "successful_topics": successful_topics,
            "failed_topics": failed_topics
        }

    async def reject_topics(self, topics_to_reject: List[str]) -> Dict:
        """Reject and delete topics from consideration."""
        rejection_results = []
        
        for topic_name in topics_to_reject:
            result = await self.qdrant_service.delete_topic(topic_name)
            rejection_results.append(result)
        
        successful_rejections = [
            result['message'] for result in rejection_results 
            if result['status'] == 'success'
        ]
        failed_rejections = [
            result['message'] for result in rejection_results 
            if result['status'] == 'error'
        ]
        
        return {
            "status": "success" if successful_rejections else "partial_error",
            "message": "Topic rejection processed",
            "successful_rejections": successful_rejections,
            "failed_rejections": failed_rejections
        }
    
  
    
 
from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from app.models.topic_models import QueryRequest, Topic, TopicResponse, InputType, TextInput
from app.services.qdrant_service import QdrantService
from app.services.topic_service import TopicService
import base64
import logging
import asyncio

router = APIRouter(prefix="/api/v1/topics")
topic_service = TopicService()
qdrant_service = QdrantService()

async def run_in_thread(func, *args, **kwargs):
    """Run a function in a separate thread."""
    return await asyncio.to_thread(func, *args, **kwargs)

@router.post("/extract-from-text", response_model=TopicResponse)
async def extract_topics_from_text(
    content: str = Form(...),
):
    """
    Extract concepts from plain text input
    """
    try:
        input_data = TextInput(
            input_type=InputType.TEXT,
            content=content
        )
        response = await topic_service.process_input(input_data)  # Await the async function
        return response
    except Exception as e:
        logging.error(f"Error processing input: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-from-url", response_model=TopicResponse)
async def extract_topics_from_url(
    url: str = Form(...)
):
    """
    Extract concepts from URL input
    """
    try:
        input_data = TextInput(
            input_type=InputType.URL,
            content=url
        )
        response = await topic_service.process_input(input_data)  # Await the async function
        return response
    except Exception as e:
        logging.error(f"Error processing input: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-from-pdf", response_model=TopicResponse)
async def extract_topics_from_pdf(
    file: UploadFile = File(...)
):
    """
    Extract concepts from PDF file including tables and images
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        contents = await file.read()
        pdf_base64 = base64.b64encode(contents).decode() 
        
        input_data = TextInput(
            input_type=InputType.PDF,
            content=pdf_base64
        )
        
        response = await topic_service.process_input(input_data)  # Await the async function
        return response
    except Exception as e:
        logging.error(f"Error processing PDF input: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/store-selected-topics", response_model=TopicResponse)
async def store_selected_topics(
    selected_topics: List[Topic]
):
    """Store selected topics in the database."""
    try:
        storage_result = await topic_service.store_selected_topics(selected_topics)  # Await the async function
        return TopicResponse(
            topics=storage_result.get("successful_topics", []),
            message=storage_result.get("message", "Processed topics"),
            status=storage_result.get("status", "similar_topics_found")
        )
    except Exception as e:
        logging.error(f"Error storing selected topics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reject-topics", response_model=TopicResponse)
async def reject_topics(
    topics_to_reject: List[str]
):
    """Reject topics and remove them from consideration."""
    try:
        rejection_result = await topic_service.reject_topics(topics_to_reject)  # Await the async function
        return TopicResponse(
            topics=[],
            message=rejection_result.get("message", "Processed topic rejections"),
            status=rejection_result.get("status", "partial_error")
        )
    except Exception as e:
        logging.error(f"Error rejecting topics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask-question", response_model=TopicResponse)
async def ask_question(request: QueryRequest):
    """
    Respond to user questions based on stored topics in the database.
    """
    query = request.query
    topics = await qdrant_service.query_topics(query)  # Await the async function
    
    if not topics:
        return TopicResponse(
            topics=[],
            message="No relevant topics found.",
            status="error"
        )
    
    return TopicResponse(
        topics=topics,
        message="Successfully fetched topics.",
        status="success"
    )

@router.get("/get-all-topics", response_model=TopicResponse)
async def get_all_topics():
    """
    Retrieve all topics from the database, sorted by most recently stored first.
    """
    try:
        topics = await qdrant_service.get_all_topics()  # Await the async function
        return TopicResponse(
            topics=topics,
            message="Successfully retrieved all topics",
            status="success"
        )
    except Exception as e:
        logging.error(f"Error retrieving topics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
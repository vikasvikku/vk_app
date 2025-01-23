from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class InputType(str, Enum):
    TEXT = "text"
    URL = "url"
    PDF = "pdf"

class TopicAttribute(BaseModel):
    field: str
    sub_field: str
    subject_matter: str
    relevance: str
    potential_impact: str
    hotness: str

class Topic(BaseModel):
    topic: str
    attributes: TopicAttribute

class TopicList(BaseModel):
    topics: List[Topic]

class TextInput(BaseModel):
    input_type: InputType
    content: str = Field(..., description="Text content, URL, or base64 encoded PDF")

class TopicResponse(BaseModel):
    topics: List[Topic]
    message: str
    status: str

class QueryRequest(BaseModel):
    query: str




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import topic_api
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(level levelname)s - %(message)s'
)

app = FastAPI(
    title="Topic Engine for Hue Ai",
    description="""Enhanced API for extracting and analyzing topics from various sources:
    Features:
    - Multiple input sources (Text, PDF, URL)
    - User interest specification
    - Topic uniqueness analysis
    - Individual topic storage control""",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(topic_api.router)
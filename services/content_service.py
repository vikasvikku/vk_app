import re
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io
import base64
from bs4 import BeautifulSoup
import requests
import logging
from typing import Tuple, List
from fastapi import HTTPException
import tabula
import asyncio

class ContentService:
    @staticmethod
    def extract_tables_from_pdf(pdf_bytes: bytes) -> List[str]:
        """Extract tables from PDF using tabula-py"""
        try:
            tables = []
            dfs = tabula.read_pdf(io.BytesIO(pdf_bytes), pages='all')
            for df in dfs:
                tables.append(df.to_string())
            return tables
        except Exception as e:
            logging.warning(f"Failed to extract tables: {str(e)}")
            return []

    @staticmethod
    async def extract_images_from_pdf(pdf_bytes: bytes) -> List[str]:
        """Extract and process images from PDF using PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            texts = []

            for page in doc:
                images = page.get_images()
                for img in images:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))

                    # Extract text from image using OCR
                    text = await asyncio.to_thread(pytesseract.image_to_string, image)
                    if text.strip():
                        texts.append(text)

            return texts
        except Exception as e:
            logging.warning(f"Failed to extract images: {str(e)}")
            return []

    @staticmethod
    async def extract_text_from_pdf(pdf_base64: str) -> Tuple[str, List[str], List[str]]:
        try:
            pdf_bytes = base64.b64decode(pdf_base64)

            # Extract text
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PdfReader(pdf_file)
            text = []

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

            # Extract tables
            tables = ContentService.extract_tables_from_pdf(pdf_bytes)

            # Extract images
            image_texts = await ContentService.extract_images_from_pdf(pdf_bytes)

            # Combine all text
            all_text = " ".join(text)

            return all_text, tables, image_texts
        except Exception as e:
            logging.error(f"Failed to process PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

    @staticmethod
    async def extract_text_from_url(url: str) -> Tuple[str, List[str]]:
        """Extract text and tables from a webpage URL."""
        try:
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()

            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                # Extract paragraphs and headings
                content_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                text = ' '.join([elem.get_text(strip=True) for elem in content_elements])
            else:
                # Fallback if no main content area found
                text = soup.get_text(strip=True)

            # Remove excessive whitespace and unwanted special characters
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,!?-]', '', text)

            # Extract tables if any
            tables = []
            for table in soup.find_all('table'):
                tables.append(table.get_text(separator=" ", strip=True))

            if not text.strip():
                raise ValueError("No readable content found on the page")

            return text.strip(), tables

        except requests.RequestException as e:
            logging.error(f"Failed to fetch URL: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing URL content: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing URL content: {str(e)}")
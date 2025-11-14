# src/metadata.py
"""
Text/image extraction, OCR, and metadata comparison.
"""

import fitz
import pytesseract
from PIL import Image
import numpy as np
import re
import logging
import io

logging.basicConfig(level=logging.INFO)

class MetadataExtractor:
    def __init__(self, pdf_path: str):
        try:
            self.doc = fitz.open(pdf_path)
        except Exception as e:
            logging.error(f"Failed to open PDF '{pdf_path}': {e}")
            self.doc = None

    def extract_text_blocks(self, page_num: int):
        if self.doc is None:
            logging.error("No PDF loaded for text extraction.")
            return {}
        try:
            page = self.doc.load_page(page_num)
            return page.get_text("dict")
        except Exception as e:
            logging.error(f"Failed to extract text blocks from page {page_num}: {e}")
            return {}

    def extract_images(self, page_num: int):
        if self.doc is None:
            logging.error("No PDF loaded for image extraction.")
            return []
        images = []
        try:
            page = self.doc.load_page(page_num)
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))
                images.append(img_pil)
            return images
        except Exception as e:
            logging.error(f"Failed to extract images from page {page_num}: {e}")
            return []

    def ocr_image(self, img_pil: Image.Image, lang: str = "eng") -> str:
        try:
            return pytesseract.image_to_string(img_pil, lang=lang)
        except Exception as e:
            logging.error(f"OCR failed: {e}")
            return ""

    def close(self):
        if self.doc is not None:
            try:
                self.doc.close()
            except Exception as e:
                logging.warning(f"Error closing PDF: {e}")

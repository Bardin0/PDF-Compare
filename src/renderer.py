# src/renderer.py
"""
PDF rasterization and image export using PyMuPDF.
"""

import fitz  # PyMuPDF
from typing import Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class PDFRenderer:

    def __init__(self, pdf_path: str):
        try:
            self.doc = fitz.open(pdf_path)
        except Exception as e:
            logging.error(f"Failed to open PDF '{pdf_path}': {e}")
            self.doc = None

    def get_page_count(self) -> int:
        if self.doc is None:
            return 0
        try:
            return self.doc.page_count
        except Exception as e:
            logging.error(f"Failed to get page count: {e}")
            return 0

    def render_page(self, page_num: int, dpi: int = 300, scale: float = 1.0) -> np.ndarray:
        if self.doc is None:
            logging.error("No PDF loaded.")
            return np.zeros((1, 1, 3), dtype=np.uint8)
        try:
            page = self.doc.load_page(page_num)
            mat = fitz.Matrix(scale, scale) * fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[..., :3]  # Drop alpha
            return img
        except Exception as e:
            logging.error(f"Failed to render page {page_num}: {e}")
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        if self.doc is not None:
            try:
                self.doc.close()
            except Exception as e:
                logging.warning(f"Error closing PDF: {e}")

# src/renderer.py
"""
PDF rasterization and image export using PyMuPDF.
"""

import fitz  # PyMuPDF
from typing import Tuple, Optional
import numpy as np
import cv2
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
        """
        Render the given page and return an OpenCV-friendly BGR uint8 ndarray.
        PyMuPDF returns RGB byte order in pix.samples; OpenCV expects BGR.
        This function ensures returned arrays are BGR (3 channels, uint8).
        """
        if self.doc is None:
            logging.error("No PDF loaded.")
            return np.zeros((1, 1, 3), dtype=np.uint8)
        try:
            page = self.doc.load_page(page_num)
            mat = fitz.Matrix(scale, scale) * fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # Build numpy array (height, width, channels)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            expected_len = pix.height * pix.width * pix.n
            if img.size != expected_len:
                # fallback reshape attempt for robustness
                img = img.copy()
            img = img.reshape((pix.height, pix.width, pix.n))
            # If there is an alpha channel, drop it
            if pix.n == 4:
                img = img[..., :3]
            # PyMuPDF returns RGB ordering; convert to BGR for OpenCV usage
            if img.ndim == 3 and img.shape[2] == 3:
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except Exception:
                    # If conversion fails for any reason, swap channels manually
                    img = img[..., ::-1].copy()
            # Ensure uint8 dtype
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
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
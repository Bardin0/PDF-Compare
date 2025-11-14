# src/diff_engine.py
"""
Diff computation and mask generation using scikit-image.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import logging

logging.basicConfig(level=logging.INFO)

class DiffEngine:
    def __init__(self, threshold: float = 0.2, morph_radius: int = 3):
        self.threshold = threshold
        self.morph_radius = morph_radius

    def compute_diff(self, imgA: np.ndarray, imgB: np.ndarray) -> np.ndarray:
        try:
            grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
            score, diff = ssim(grayA, grayB, full=True)
            diff = (1.0 - diff).astype(np.float32)
            mask = (diff > self.threshold).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_radius, self.morph_radius))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            return mask
        except Exception as e:
            logging.error(f"Exception during diff computation: {e}")
            return np.zeros((imgA.shape[0], imgA.shape[1]), dtype=np.uint8)

# src/register.py
"""
Image alignment and registration using OpenCV.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)

class PageRegister:
    def __init__(self):
        self.detector = cv2.ORB_create(5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align(self, imgA: np.ndarray, imgB: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            # If images are identical, return identity transform
            if np.array_equal(imgA, imgB):
                H = np.eye(3, dtype=np.float32)
                return imgB.copy(), H
            grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
            kpA, desA = self.detector.detectAndCompute(grayA, None)
            kpB, desB = self.detector.detectAndCompute(grayB, None)
            if desA is None or desB is None:
                logging.warning("No features detected in one or both images for alignment.")
                return imgB, None
            matches = self.matcher.match(desA, desB)
            if len(matches) < 10:
                logging.warning(f"Not enough matches for alignment: {len(matches)} found.")
                return imgB, None
            src_pts = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if H is not None:
                aligned = cv2.warpPerspective(imgB, H, (imgA.shape[1], imgA.shape[0]))
                return aligned, H
            logging.warning("Homography estimation failed.")
            return imgB, None
        except Exception as e:
            logging.error(f"Exception during alignment: {e}")
            return imgB, None

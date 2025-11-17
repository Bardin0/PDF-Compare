from PySide6.QtCore import QObject, Signal
import cv2
import numpy as np
from src.gui import PDFDiffViewer

class CompareWorker(QObject):
    finished = Signal(list, list)
    error = Signal(str)

    def __init__(self, rendererA, rendererB, dpi, register, diff_engine):
        super().__init__()
        self.rendererA = rendererA
        self.rendererB = rendererB
        self.dpi = dpi
        self.register = register
        self.diff_engine = diff_engine

    def run(self):
        try:
            diffs = []
            bboxes_all = []
            n_pages = min(self.rendererA.get_page_count(),
                          self.rendererB.get_page_count())

            for i in range(n_pages):
                imgA = self.rendererA.render_page(i, dpi=self.dpi)
                imgB = self.rendererB.render_page(i, dpi=self.dpi)
                alignedB, _ = self.register.align(imgA, imgB)
                mask = self.diff_engine.compute_diff(imgA, alignedB)

                diff_img = PDFDiffViewer.overlay_mask(imgA, mask)

                # bounding boxes
                kernel = np.ones((7, 7), np.uint8)
                closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 20]
                bboxes.sort(key=lambda r: (r[1], r[0]))

                diffs.append(diff_img)
                bboxes_all.append(bboxes)

            self.finished.emit(diffs, bboxes_all)

        except Exception as e:
            self.error.emit(str(e))

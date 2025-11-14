# src/gui.py
"""
Main PySide6 application and synchronized views.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel, QSlider, QMessageBox, QScrollArea
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import numpy as np
import cv2
import logging
from src.renderer import PDFRenderer
from src.register import PageRegister
from src.diff_engine import DiffEngine

logging.basicConfig(level=logging.INFO)

class PDFDiffViewer(QMainWindow):
    @staticmethod
    def overlay_mask(img, mask):
        import cv2
        import numpy as np
        overlay = img.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        # Highlight mask in red
        overlay[mask > 0] = [255, 0, 0]
        return overlay

    def scale_to_label(self, img, label):
        # Scale image to fill label area while maintaining aspect ratio
        import cv2
        label_width = label.width() if label.width() > 0 else 350
        label_height = label.height() if label.height() > 0 else 600
        h, w = img.shape[:2]
        scale = min(label_width / w, label_height / h)
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def np_to_pixmap(img):
        # Convert a NumPy array to QPixmap
        import numpy as np
        from PySide6.QtGui import QImage, QPixmap
        if img is None:
            return QPixmap()
        if img.dtype != np.uint8:
            img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
        if img.ndim == 2:
            # Grayscale
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        elif img.ndim == 3 and img.shape[2] == 3:
            # RGB
            h, w, ch = img.shape
            qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        elif img.ndim == 3 and img.shape[2] == 4:
            # RGBA
            h, w, ch = img.shape
            qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Unsupported image shape for QPixmap conversion: {img.shape}")
        return QPixmap.fromImage(qimg.copy())
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Image Diff Viewer")
        self.setFixedSize(1200, 800)  # Fixed window size
        self._init_ui()
        self.rendererA = None
        self.rendererB = None
        self.pageA = 0
        self.pageB = 0
        self.page_countA = 1
        self.page_countB = 1
        self.dpi = 150
        self.zoom_factor = 1.0  # Initial zoom factor
        self.register = PageRegister()
        self.diff_engine = DiffEngine()
        self.diffs = []  # Store computed diffs for all pages

    def _init_ui(self):
        central = QWidget()
        layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.loadA_btn = QPushButton("Load PDF A")
        self.loadB_btn = QPushButton("Load PDF B")
        self.compare_btn = QPushButton("Compare")
        btn_layout.addWidget(self.loadA_btn)
        btn_layout.addWidget(self.loadB_btn)
        btn_layout.addWidget(self.compare_btn)
        layout.addLayout(btn_layout)

        # Unified page navigation controls
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Previous Page")
        self.next_btn = QPushButton("Next Page >")
        self.page_label = QLabel("Page: 1")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.page_label)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

        # Side-by-side layout for PDF A, PDF B, and Diff
        img_layout = QHBoxLayout()
        self.imgA_label = ZoomLabel("PDF A", self, 'A')
        self.imgA_label.setAlignment(Qt.AlignCenter)
        self.imgA_label.setMinimumSize(350, 600)
        self.imgA_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.imgB_label = ZoomLabel("PDF B", self, 'B')
        self.imgB_label.setAlignment(Qt.AlignCenter)
        self.imgB_label.setMinimumSize(350, 600)
        self.imgB_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.diff_label = ZoomLabel("Diff", self, 'D')
        self.diff_label.setAlignment(Qt.AlignCenter)
        self.diff_label.setMinimumSize(350, 600)
        self.diff_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        img_layout.addWidget(self.imgA_label)
        img_layout.addWidget(self.imgB_label)
        img_layout.addWidget(self.diff_label)
        layout.addLayout(img_layout)

        central.setLayout(layout)
        self.setCentralWidget(central)
        self.loadA_btn.clicked.connect(self.load_pdf_a)
        self.loadB_btn.clicked.connect(self.load_pdf_b)
        self.compare_btn.clicked.connect(self.compare)
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)

    def load_pdf_a(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open PDF A", "", "PDF Files (*.pdf)")
            if path:
                self.rendererA = PDFRenderer(path)
                self.pageA = 0
                self.page_countA = self.rendererA.get_page_count()
                self.show_page_a()
        except Exception as e:
            logging.error(f"Failed to load PDF A: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load PDF A: {e}")

    def load_pdf_b(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open PDF B", "", "PDF Files (*.pdf)")
            if path:
                self.rendererB = PDFRenderer(path)
                self.pageB = 0
                self.page_countB = self.rendererB.get_page_count()
                self.show_page_b()
        except Exception as e:
            logging.error(f"Failed to load PDF B: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load PDF B: {e}")

    def show_page_a(self):
        try:
            if self.rendererA:
                img = self.rendererA.render_page(self.pageA, dpi=int(self.dpi * self.zoom_factor))
                pixmap = self.np_to_pixmap(self.scale_to_label(img, self.imgA_label))
                self.imgA_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Failed to render PDF A page: {e}")
            QMessageBox.critical(self, "Error", f"Failed to render PDF A page: {e}")

    def show_page_b(self):
        try:
            if self.rendererB:
                img = self.rendererB.render_page(self.pageB, dpi=int(self.dpi * self.zoom_factor))
                pixmap = self.np_to_pixmap(self.scale_to_label(img, self.imgB_label))
                self.imgB_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Failed to render PDF B page: {e}")
            QMessageBox.critical(self, "Error", f"Failed to render PDF B page: {e}")
    def prev_page(self):
        # Move both PDFs to previous page (if possible), update only page and images
        if self.rendererA and self.pageA > 0:
            self.pageA -= 1
        if self.rendererB and self.pageB > 0:
            self.pageB -= 1
        self.show_page_a()
        self.show_page_b()
        self.show_diff()
        self.update_page_label()

    def next_page(self):
        # Move both PDFs to next page (if possible), update only page and images
        if self.rendererA and self.pageA < self.page_countA - 1:
            self.pageA += 1
        if self.rendererB and self.pageB < self.page_countB - 1:
            self.pageB += 1
        self.show_page_a()
        self.show_page_b()
        self.show_diff()
        self.update_page_label()

    def compare(self):
        try:
            self.diffs = []
            if self.rendererA and self.rendererB:
                countA = self.rendererA.get_page_count()
                countB = self.rendererB.get_page_count()
                n_pages = min(countA, countB)
                for i in range(n_pages):
                    imgA = self.rendererA.render_page(i, dpi=self.dpi)
                    imgB = self.rendererB.render_page(i, dpi=self.dpi)
                    alignedB, _ = self.register.align(imgA, imgB)
                    mask = self.diff_engine.compute_diff(imgA, alignedB)
                    diff_img = self.overlay_mask(imgA, mask)
                    self.diffs.append(diff_img)
                self.show_diff()
        except Exception as e:
            logging.error(f"Failed to compute all diffs: {e}")
            QMessageBox.critical(self, "Error", f"Failed to compute all diffs: {e}")

    def show_diff(self):
        # Show only the diff for the current page
        if self.diffs and 0 <= self.pageA < len(self.diffs):
            diff_img = self.diffs[self.pageA]
            pixmap = self.np_to_pixmap(self.scale_to_label(diff_img, self.diff_label))
            self.diff_label.setPixmap(pixmap)
        else:
            self.diff_label.clear()

    def set_zoom(self, factor):
        # Clamp zoom factor
        self.zoom_factor = max(0.2, min(factor, 5.0))
        self.show_page_a()
        self.show_page_b()
        self.show_diff()
# Custom QLabel to handle wheel events for zoom
class ZoomLabel(QLabel):
    def __init__(self, text, parent, which):
        super().__init__(text)
        self.parent = parent
        self.which = which  # 'A', 'B', or 'D'

    def wheelEvent(self, event):
        # Zoom in/out on wheel, synchronize all
        delta = event.angleDelta().y()
        if delta > 0:
            self.parent.set_zoom(self.parent.zoom_factor * 1.1)
        elif delta < 0:
            self.parent.set_zoom(self.parent.zoom_factor / 1.1)

    def update_page_label(self):
        # Update the page label to show current page and total
        max_pages = max(self.page_countA, self.page_countB)
        self.page_label.setText(f"Page: {self.pageA+1} / {max_pages}")
    def scale_to_label(self, img: np.ndarray, label: QLabel) -> np.ndarray:
        # Scale image to fill label area while maintaining aspect ratio
        label_width = label.width() if label.width() > 0 else 350
        label_height = label.height() if label.height() > 0 else 600
        h, w = img.shape[:2]
        scale = min(label_width / w, label_height / h)
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    # Removed update_zoom and slider functionality

    # Removed duplicate/incorrect np_to_pixmap

    @staticmethod
    def overlay_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = img.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        overlay[mask > 0] = [255, 0, 0]
        return overlay

def main():
    app = QApplication(sys.argv)
    viewer = PDFDiffViewer()
    viewer.show()
    sys.exit(app.exec())

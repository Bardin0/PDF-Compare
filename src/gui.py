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
    def overlay_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = img.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        overlay[mask > 0] = [255, 0, 0]
        return overlay

    def scale_to_label(self, img, label, zoom=1.0):
        # Scale image to fill label area while maintaining aspect ratio, then apply zoom and pan
        label_width = label.width() if label.width() > 0 else 350
        label_height = label.height() if label.height() > 0 else 600
        h, w = img.shape[:2]
        scale = min(label_width / w, label_height / h) * zoom
        # Pan offset
        pan = getattr(label, '_pan', [0, 0])
        # Resize
        if scale != 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        # Crop to label size, centered and panned
        img_h, img_w = img.shape[:2]
        cx = img_w // 2 - int(pan[0] * scale)
        cy = img_h // 2 - int(pan[1] * scale)
        x0 = max(0, cx - label_width // 2)
        y0 = max(0, cy - label_height // 2)
        x1 = min(img_w, x0 + label_width)
        y1 = min(img_h, y0 + label_height)
        cropped = np.zeros((label_height, label_width, 3), dtype=img.dtype) if img.ndim == 3 else np.zeros((label_height, label_width), dtype=img.dtype)
        crop = img[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        cropped[0:ch, 0:cw] = crop
        return cropped

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
        self.resize(1200, 800)  # Initial size, but resizable
        self.setMinimumSize(800, 600)
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

        # Add full screen toggle shortcut (F11)
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
        self._is_fullscreen = False

    def toggle_fullscreen(self):
        if self._is_fullscreen:
            self.showNormal()
            self._is_fullscreen = False
        else:
            self.showFullScreen()
            self._is_fullscreen = True

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

        # Zoom controls
        zoom_layout = QHBoxLayout()
        from PySide6.QtWidgets import QSlider, QLineEdit
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)  # 0.1x
        self.zoom_slider.setMaximum(1000) # 10.0x
        # Ensure zoom_factor is initialized before use
        if not hasattr(self, 'zoom_factor'):
            self.zoom_factor = 1.0
        self.zoom_slider.setValue(int(self.zoom_factor * 100))
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setSingleStep(1)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider)

        self.zoom_input = QLineEdit(str(self.zoom_factor))
        self.zoom_input.setFixedWidth(50)
        self.zoom_input.editingFinished.connect(self._on_zoom_input)

        zoom_layout.addWidget(QLabel("Zoom:"))
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_input)
        layout.addLayout(zoom_layout)

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

    def update_page_label(self):
        # Update the page label to show current page and total
        max_pages = max(getattr(self, 'page_countA', 1), getattr(self, 'page_countB', 1))
        self.page_label.setText(f"Page: {getattr(self, 'pageA', 0)+1} / {max_pages}")

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
                img = self.rendererA.render_page(self.pageA, dpi=self.dpi)
                img = self.scale_to_label(img, self.imgA_label, self.zoom_factor)
                pixmap = self.np_to_pixmap(img)
                self.imgA_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Failed to render PDF A page: {e}")
            QMessageBox.critical(self, "Error", f"Failed to render PDF A page: {e}")

    def show_page_b(self):
        try:
            if self.rendererB:
                img = self.rendererB.render_page(self.pageB, dpi=self.dpi)
                img = self.scale_to_label(img, self.imgB_label, self.zoom_factor)
                pixmap = self.np_to_pixmap(img)
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
            diff_img = self.scale_to_label(diff_img, self.diff_label, self.zoom_factor)
            pixmap = self.np_to_pixmap(diff_img)
            self.diff_label.setPixmap(pixmap)
        else:
            self.diff_label.clear()

    def set_zoom(self, factor, center=None, update_controls=True):
        # Clamp zoom factor
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(factor, 10.0))
        # Adjust pan to keep mouse position stable if center is given
        if center is not None:
            for label in [self.imgA_label, self.imgB_label, self.diff_label]:
                label.adjust_pan_for_zoom(center, old_zoom, self.zoom_factor)
        # Update slider and input if needed
        if update_controls:
            # Synchronize slider and input, block signals to prevent recursion
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(int(self.zoom_factor * 100))
            self.zoom_slider.blockSignals(False)
            self.zoom_input.blockSignals(True)
            self.zoom_input.setText(f"{self.zoom_factor:.2f}")
            self.zoom_input.blockSignals(False)
        self.show_page_a()
        self.show_page_b()
        self.show_diff()

    def _on_zoom_slider(self, value):
        # Slider value is 10-1000, representing 0.1x to 10.0x
        zoom = value / 100.0
        # When slider changes, update zoom and input field
        self.set_zoom(zoom, update_controls=True)

    def _on_zoom_input(self):
        try:
            zoom = float(self.zoom_input.text())
        except Exception:
            zoom = self.zoom_factor
        # When input changes, update zoom and slider
        self.set_zoom(zoom, update_controls=True)
# Custom QLabel to handle wheel events for zoom
class ZoomLabel(QLabel):
    def __init__(self, text, parent, which):
        super().__init__(text)
        self.parent = parent
        self.which = which  # 'A', 'B', or 'D'
        self.setMouseTracking(True)
        self._drag_active = False
        self._last_pos = None
        self._pan = [0, 0]  # [x, y] pan offset in image coordinates

    def wheelEvent(self, event):
        # Zoom in/out on wheel, synchronize all, centered on mouse
        delta = event.angleDelta().y()
        # Faster zoom: 1.25x per step, allow up to 10.0x
        zoom_step = 1.25
        mouse_pos = event.position() if hasattr(event, 'position') else event.posF() if hasattr(event, 'posF') else event.pos()
        center = (mouse_pos.x(), mouse_pos.y())
        if delta > 0:
            self.parent.set_zoom(min(self.parent.zoom_factor * zoom_step, 10.0), center=center)
        elif delta < 0:
            self.parent.set_zoom(max(self.parent.zoom_factor / zoom_step, 0.1), center=center)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_active and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            # Pan in image coordinates, scale by zoom
            dx = int(-delta.x() / self.parent.zoom_factor)
            dy = int(-delta.y() / self.parent.zoom_factor)
            for label in [self.parent.imgA_label, self.parent.imgB_label, self.parent.diff_label]:
                label._pan[0] += dx
                label._pan[1] += dy
            self.parent.show_page_a()
            self.parent.show_page_b()
            self.parent.show_diff()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            self._last_pos = None

    def adjust_pan_for_zoom(self, center, old_zoom, new_zoom):
        # Adjust pan so that the point under the mouse stays fixed
        if center is None:
            return
        label_width = self.width() if self.width() > 0 else 350
        label_height = self.height() if self.height() > 0 else 600
        x, y = center
        # Convert label coords to image coords at old zoom
        img_x = (x - label_width / 2) / old_zoom - self._pan[0]
        img_y = (y - label_height / 2) / old_zoom - self._pan[1]
        # After zoom, what pan is needed to keep this point under the mouse?
        new_pan_x = (x - label_width / 2) / new_zoom - img_x
        new_pan_y = (y - label_height / 2) / new_zoom - img_y
        self._pan = [new_pan_x, new_pan_y]





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

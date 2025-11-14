# src/gui.py
"""
Main PySide6 application and synchronized views.
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFileDialog, QLabel, QSlider, QMessageBox,
    QLineEdit, QToolBar, QSizePolicy, QToolButton, QMenu
)
from PySide6.QtGui import QShortcut, QKeySequence, QAction
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
        """
        Overlay a bright yellow highlight with padding on the diff mask.
        Image arrays are BGR (OpenCV). Bright yellow in BGR is (0,255,255).
        """
        overlay = img.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        # Dilate the mask to add padding around differences
        kernel = np.ones((15, 15), np.uint8)  # Padding size (adjust as needed)
        padded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        # Create a yellow highlight (semi-transparent)
        yellow_bgr = np.array([0, 255, 255], dtype=np.uint8)  # BGR for yellow
        alpha = 0.5  # Transparency
        highlight = np.zeros_like(overlay, dtype=np.uint8)
        highlight[:, :] = yellow_bgr
        # Blend highlight where padded_mask is set
        mask_indices = padded_mask > 0
        # cv2.addWeighted works on arrays of same shape; index returns N x 3 arrays
        if mask_indices.ndim == 2:
            try:
                overlay[mask_indices] = cv2.addWeighted(
                    overlay[mask_indices], 1.0 - alpha, highlight[mask_indices], alpha, 0
                )
            except Exception:
                # Fallback: assign color directly if blend fails
                overlay[mask_indices] = yellow_bgr
        else:
            overlay[mask_indices] = yellow_bgr
        return overlay

    def scale_to_label(self, img, label, zoom=1.0):
        # Scale image to fill label area while maintaining aspect ratio, then apply zoom and pan
        label_width = label.width() if label.width() > 0 else 350
        label_height = label.height() if label.height() > 0 else 600
        h, w = img.shape[:2]
        scale = min(label_width / w, label_height / h) * zoom
        # Pan offset (in image pixel coordinates)
        pan = getattr(label, '_pan', [0.0, 0.0])
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
        if img.ndim == 3:
            cropped = np.zeros((label_height, label_width, 3), dtype=img.dtype)
        else:
            cropped = np.zeros((label_height, label_width), dtype=img.dtype)
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
            # safe normalization
            mn = float(img.min())
            mx = float(img.max())
            if mx - mn == 0:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = (255 * (img - mn) / (mx - mn)).astype(np.uint8)
        if img.ndim == 2:
            # Grayscale
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        elif img.ndim == 3 and img.shape[2] == 3:
            # BGR -> RGB for display
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        elif img.ndim == 3 and img.shape[2] == 4:
            # BGRA -> RGBA
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            h, w, ch = rgba.shape
            qimg = QImage(rgba.data, w, h, ch * w, QImage.Format_RGBA8888)
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
        self.diff_bboxes = []  # bounding boxes per page
        self.current_diff_idx = 0

        # Add full screen toggle shortcut (F11)
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
        # Central widget and main layout
        central = QWidget()
        main_layout = QVBoxLayout()

        # Toolbar setup: all actions in one horizontal line
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # File menu as QToolButton
        file_menu = QMenu("File", self)
        self.action_loadA = QAction("Upload PDF A", self)
        self.action_loadB = QAction("Upload PDF B", self)
        self.action_export = QAction("Export Compared PDF", self)
        self.action_export.setEnabled(False)
        file_menu.addAction(self.action_loadA)
        file_menu.addAction(self.action_loadB)
        file_menu.addSeparator()
        file_menu.addAction(self.action_export)
        file_button = QToolButton()
        file_button.setText("File")
        file_button.setMenu(file_menu)
        file_button.setPopupMode(QToolButton.InstantPopup)
        file_button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; height: 0; }")
        toolbar.addWidget(file_button)

        # View menu as QToolButton
        view_menu = QMenu("View", self)
        self.action_next_diff = QAction("Next Diff (Alt+→)", self)
        self.action_prev_diff = QAction("Previous Diff (Alt+←)", self)
        self.action_reset_view = QAction("Reset View (Alt+↓)", self)
        view_menu.addAction(self.action_next_diff)
        view_menu.addAction(self.action_prev_diff)
        view_menu.addSeparator()
        view_menu.addAction(self.action_reset_view)
        view_button = QToolButton()
        view_button.setText("View")
        view_button.setMenu(view_menu)
        view_button.setPopupMode(QToolButton.InstantPopup)
        view_button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; height: 0; }")
        toolbar.addWidget(view_button)

        # Navigation/compare actions as toolbar buttons
        self.action_compare = QAction("Compare", self)
        self.action_prev = QAction("Previous Page (Ctrl+←)", self)
        self.action_next = QAction("Next Page (Ctrl+→)", self)
        toolbar.addAction(self.action_compare)
        toolbar.addAction(self.action_prev)
        toolbar.addAction(self.action_next)

        # Side-by-side layout for PDF A, PDF B, and Diff (equal, flexible)
        img_layout = QHBoxLayout()
        self.imgA_label = ZoomLabel("PDF A", self, 'A')
        self.imgA_label.setAlignment(Qt.AlignCenter)
        self.imgA_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.imgA_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.imgB_label = ZoomLabel("PDF B", self, 'B')
        self.imgB_label.setAlignment(Qt.AlignCenter)
        self.imgB_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.imgB_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.diff_label = ZoomLabel("Diff", self, 'D')
        self.diff_label.setAlignment(Qt.AlignCenter)
        self.diff_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.diff_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_layout.addWidget(self.imgA_label, stretch=1)
        img_layout.addWidget(self.imgB_label, stretch=1)
        img_layout.addWidget(self.diff_label, stretch=1)
        main_layout.addLayout(img_layout)

        # Bottom bar: page label (left), zoom controls (right)
        bottom_bar = QHBoxLayout()
        self.page_label = QLabel("Page: 1")
        self.page_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bottom_bar.addWidget(self.page_label)
        bottom_bar.addStretch(1)
        zoom_label = QLabel("Zoom:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)  # 0.1x
        self.zoom_slider.setMaximum(1000)  # 10.0x
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
        bottom_bar.addWidget(zoom_label)
        bottom_bar.addWidget(self.zoom_slider)
        bottom_bar.addWidget(self.zoom_input)
        main_layout.addLayout(bottom_bar)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Connect actions
        self.action_loadA.triggered.connect(self.load_pdf_a)
        self.action_loadB.triggered.connect(self.load_pdf_b)
        self.action_export.triggered.connect(self.export_compared_pdf)
        self.action_compare.triggered.connect(self.compare)
        self.action_prev.triggered.connect(self.prev_page)
        self.action_next.triggered.connect(self.next_page)
        self.action_next_diff.triggered.connect(lambda: self.goto_diff(1))
        self.action_prev_diff.triggered.connect(lambda: self.goto_diff(-1))
        self.action_reset_view.triggered.connect(self.reset_view)  # Connect reset action

        # Add keyboard shortcuts for diff navigation
        # Add keyboard shortcuts for page navigation
        QShortcut(QKeySequence("Ctrl+Right"), self, self.next_page)
        QShortcut(QKeySequence("Ctrl+Left"), self, self.prev_page)
        QShortcut(QKeySequence("Alt+Right"), self, lambda: self.goto_diff(1))
        QShortcut(QKeySequence("Alt+Left"), self, lambda: self.goto_diff(-1))
        QShortcut(QKeySequence("Alt+Down"), self, self.reset_view)

    def export_compared_pdf(self):
        """
        Export the compared PDF (diff-highlighted images) as a new PDF file.
        """
        if not self.diffs:
            QMessageBox.warning(self, "No Diff", "Please compare PDFs first.")
            return
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Save Compared PDF", "", "PDF Files (*.pdf)")
            if not path:
                return
            import fitz  # PyMuPDF
            import cv2
            import numpy as np
            doc = fitz.open()
            for img in self.diffs:
                # Convert to RGB if needed
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("Unsupported image shape for export")
                height, width = img.shape[:2]
                # Encode as PNG in memory
                success, png_bytes = cv2.imencode('.png', img)
                if not success:
                    raise RuntimeError("Failed to encode image as PNG for PDF export")
                png_bytes = png_bytes.tobytes()
                # Create PDF page and insert image from PNG stream
                page = doc.new_page(width=width, height=height)
                page.insert_image(page.rect, stream=png_bytes)
            doc.save(path)
            doc.close()
            QMessageBox.information(self, "Export Complete", f"Compared PDF saved to:\n{path}")
        except Exception as e:
            logging.error(f"Failed to export compared PDF: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export compared PDF: {e}")

    def update_page_label(self):
        # Update the page label to show current page and total
        max_pages = max(getattr(self, 'page_countA', 1), getattr(self, 'page_countB', 1))
        self.page_label.setText(f"Page: {getattr(self, 'pageA', 0) + 1} / {max_pages}")
        # Enable export only if diffs exist
        if hasattr(self, 'action_export'):
            self.action_export.setEnabled(bool(self.diffs))

    def _reset_view_state(self):
        # Reset pan and zoom state on all panels
        for panel in [self.imgA_label, self.imgB_label, self.diff_label]:
            panel._pan = [0.0, 0.0]
        self.zoom_factor = 1.0
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(self.zoom_factor * 100))
        self.zoom_slider.blockSignals(False)
        self.zoom_input.blockSignals(True)
        self.zoom_input.setText(f"{self.zoom_factor:.2f}")
        self.zoom_input.blockSignals(False)

    def reset_view(self):
        """
        Reset PDFs to their default location and zoom.
        """
        self._reset_view_state()
        # re-render current pages at default view
        self.show_page_a()
        self.show_page_b()
        self.show_diff()

    def load_pdf_a(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open PDF A", "", "PDF Files (*.pdf)")
            if path:
                self.rendererA = PDFRenderer(path)
                self.pageA = 0
                self.page_countA = self.rendererA.get_page_count()
                self._reset_view_state()
                self.show_page_a()
                # Reset diff state
                self.diffs = []
                self.diff_bboxes = []
                self.current_diff_idx = 0
                if hasattr(self, 'action_export'):
                    self.action_export.setEnabled(False)
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
                self._reset_view_state()
                self.show_page_b()
                # Reset diff state
                self.diffs = []
                self.diff_bboxes = []
                self.current_diff_idx = 0
                if hasattr(self, 'action_export'):
                    self.action_export.setEnabled(False)
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
        # Reset diff navigation for new page
        self.current_diff_idx = 0
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
        # Reset diff navigation for new page
        self.current_diff_idx = 0
        self.show_page_a()
        self.show_page_b()
        self.show_diff()
        self.update_page_label()

    def compare(self):
        try:
            self.diffs = []
            self.diff_bboxes = []
            self.current_diff_idx = 0
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
                    # --- Improved diff region detection ---
                    # Morphological closing to merge nearby regions
                    kernel = np.ones((7, 7), np.uint8)
                    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                    # Find bounding boxes of significant diff regions
                    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 20]
                    # sort bboxes left-to-right, top-to-bottom for predictable navigation
                    bboxes.sort(key=lambda r: (r[1], r[0]))
                    self.diff_bboxes.append(bboxes)
                self.show_diff()
            # Enable export if diffs exist
            if hasattr(self, 'action_export'):
                self.action_export.setEnabled(bool(self.diffs))
        except Exception as e:
            logging.error(f"Failed to compute all diffs: {e}")
            QMessageBox.critical(self, "Error", f"Failed to compute all diffs: {e}")

    def goto_diff(self, direction=1):
        """
        Navigate to the next/previous diff region on the current page, zooming to fit tightly.
        direction: 1 for next, -1 for previous
        """
        if not (self.diffs and self.pageA < len(self.diff_bboxes)):
            return
        bboxes = self.diff_bboxes[self.pageA]
        if not bboxes:
            return
        # Update index and clamp
        self.current_diff_idx = (self.current_diff_idx + direction) % len(bboxes)
        x, y, w_box, h_box = bboxes[self.current_diff_idx]

        # Add a small margin (5% of bbox size, at least 5px)
        margin_x = max(int(w_box * 0.05), 5)
        margin_y = max(int(h_box * 0.05), 5)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = x + w_box + margin_x
        y1 = y + h_box + margin_y
        bbox_w = x1 - x0
        bbox_h = y1 - y0

        # Get the original diff image (same size as source image)
        diff_img = self.diffs[self.pageA]
        img_h, img_w = diff_img.shape[:2]

        # Compute zoom to fit bbox tightly in view (based on image coords -> label fit)
        label = self.imgA_label  # labels are same size layout-wise
        label_width = label.width() if label.width() > 0 else 350
        label_height = label.height() if label.height() > 0 else 600

        # compute base scale (fit-to-label) then compute additional zoom so bbox fills label
        base_scale_x = label_width / img_w
        base_scale_y = label_height / img_h
        base_scale = min(base_scale_x, base_scale_y)
        # required final_scale to make bbox fill label:
        req_scale_x = label_width / bbox_w
        req_scale_y = label_height / bbox_h
        req_final_scale = min(req_scale_x, req_scale_y)
        # derive target zoom_factor = req_final_scale / base_scale
        if base_scale <= 0:
            target_zoom = 1.0
        else:
            target_zoom = req_final_scale / base_scale
        # clamp zoom
        target_zoom = max(0.1, min(target_zoom, 10.0))

        # Compute bbox center in image coordinates
        cx_img = x0 + bbox_w / 2.0
        cy_img = y0 + bbox_h / 2.0

        # Compute pan such that center point ends up in the center of the label.
        pan_x = (img_w / 2.0) - cx_img
        pan_y = (img_h / 2.0) - cy_img

        # Apply pan to all panels (image-coordinates style)
        for panel in [self.imgA_label, self.imgB_label, self.diff_label]:
            panel._pan = [pan_x, pan_y]

        # Apply zoom (this will update controls and redraw)
        self.set_zoom(target_zoom, center=None, update_controls=True)

        # Optionally show a short-lived visual indicator: draw bbox on diff view
        try:
            disp = diff_img.copy()
            # draw bbox in red on image coordinate space
            x0i = int(x0); y0i = int(y0); x1i = int(x1); y1i = int(y1)
            cv2.rectangle(disp, (x0i, y0i), (x1i, y1i), (0, 0, 255), 2)
            disp_scaled = self.scale_to_label(disp, self.diff_label, self.zoom_factor)
            pixmap = self.np_to_pixmap(disp_scaled)
            self.diff_label.setPixmap(pixmap)
        except Exception:
            pass

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
        # When slider changes, update zoom and input field, zoom from center
        self.set_zoom(zoom, center=None, update_controls=True)

    def _on_zoom_input(self):
        try:
            zoom = float(self.zoom_input.text())
        except Exception:
            zoom = self.zoom_factor
        # When input changes, update zoom and slider, zoom from center
        self.set_zoom(zoom, center=None, update_controls=True)


# Custom QLabel to handle wheel events for zoom
class ZoomLabel(QLabel):
    def __init__(self, text, parent, which):
        super().__init__(text)
        self.parent = parent
        self.which = which  # 'A', 'B', or 'D'
        self.setMouseTracking(True)
        self._drag_active = False
        self._last_pos = None
        self._pan = [0.0, 0.0]  # [x, y] pan offset in image coordinates

    def wheelEvent(self, event):
        """
        Zoom only when the mouse is over this label. Zoom towards the exact cursor
        location on the hovered PDF; compute corresponding points for the other panels
        so they zoom in a synchronized manner toward the same visual spot.
        """
        delta = event.angleDelta().y()
        if delta == 0:
            return

        # High-resolution position where wheel event occurred (local coords)
        try:
            posf = event.position()  # QPointF in Qt6
            pos_point = posf.toPoint()
            local_x = posf.x()
            local_y = posf.y()
        except Exception:
            # Fallback for older event API
            pos_point = event.pos()
            local_x = pos_point.x()
            local_y = pos_point.y()

        # Only zoom when cursor is inside this widget bounds
        if not (0 <= local_x < self.width() and 0 <= local_y < self.height()):
            return

        # Global position for mapping to other panels
        global_pos = self.mapToGlobal(pos_point)

        zoom_step = 1.25
        old_zoom = self.parent.zoom_factor
        if delta > 0:
            new_zoom = min(old_zoom * zoom_step, 10.0)
        else:
            new_zoom = max(old_zoom / zoom_step, 0.1)

        # If no zoom change, skip
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        # Compute per-panel centers: map the global cursor into each panel's local coords.
        panels = [self.parent.imgA_label, self.parent.imgB_label, self.parent.diff_label]
        centers = []
        for panel in panels:
            local = panel.mapFromGlobal(global_pos)
            if 0 <= local.x() < panel.width() and 0 <= local.y() < panel.height():
                centers.append((local.x(), local.y()))
            else:
                # If cursor not over that panel, use its center so the visual stays coherent.
                centers.append((panel.width() // 2, panel.height() // 2))

        # Apply the zoom and adjust pan for each panel so the point under the cursor remains stable.
        self.parent.zoom_factor = new_zoom
        for panel, center in zip(panels, centers):
            panel.adjust_pan_for_zoom(center, old_zoom, new_zoom)

        # Update slider and input
        self.parent.zoom_slider.blockSignals(True)
        self.parent.zoom_slider.setValue(int(self.parent.zoom_factor * 100))
        self.parent.zoom_slider.blockSignals(False)
        self.parent.zoom_input.blockSignals(True)
        self.parent.zoom_input.setText(f"{self.parent.zoom_factor:.2f}")
        self.parent.zoom_input.blockSignals(False)

        # Redraw
        self.parent.show_page_a()
        self.parent.show_page_b()
        self.parent.show_diff()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_active and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            # Increase speed (e.g., 2x) and invert directionality
            speed = 2.0
            dx = int(delta.x() * speed / max(self.parent.zoom_factor, 1e-6))
            dy = int(delta.y() * speed / max(self.parent.zoom_factor, 1e-6))
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
        # Adjust pan so the point under the mouse stays under the mouse after zoom
        if center is None:
            return
        label_width = self.width() if self.width() > 0 else 350
        label_height = self.height() if self.height() > 0 else 600
        x, y = center
        # Convert label coords to image coords at old zoom
        img_x = (x - label_width / 2.0) / old_zoom - self._pan[0]
        img_y = (y - label_height / 2.0) / old_zoom - self._pan[1]
        # After zoom, set pan so that this image point is exactly under the mouse
        new_pan_x = (x - label_width / 2.0) / new_zoom - img_x
        new_pan_y = (y - label_height / 2.0) / new_zoom - img_y
        self._pan[0] = new_pan_x
        self._pan[1] = new_pan_y

    @staticmethod
    def overlay_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Keep this helper aligned with main overlay color (yellow BGR)
        """
        overlay = img.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        yellow_bgr = np.array([0, 255, 255], dtype=np.uint8)
        overlay[mask > 0] = yellow_bgr
        return overlay


def main():
    app = QApplication(sys.argv)
    viewer = PDFDiffViewer()
    viewer.show()
    sys.exit(app.exec())

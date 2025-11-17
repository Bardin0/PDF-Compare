# src/gui.py
"""
Main PySide6 application and synchronized views.
Unified single pan and zoom for all three PDF views.
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFileDialog, QLabel, QSlider, QMessageBox,
    QLineEdit, QToolBar, QSizePolicy, QToolButton, QMenu
)
from PySide6.QtGui import QShortcut, QKeySequence, QAction, QActionGroup
from PySide6.QtCore import Qt, QThread

import numpy as np
import cv2
import logging
from src.renderer import PDFRenderer
from src.register import PageRegister
from src.diff_engine import DiffEngine
from src.spinner import LoadingSpinner

logging.basicConfig(level=logging.INFO)


class PDFDiffViewer(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_mode = "highlight"
        self.slider_temp_diff_rect = None  # (x, y, w, h) in image coords for slider highlight
        # ...existing code...
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
        if mask_indices.ndim == 2:
            try:
                overlay[mask_indices] = cv2.addWeighted(
                    overlay[mask_indices], 1.0 - alpha, highlight[mask_indices], alpha, 0
                )
            except Exception:
                overlay[mask_indices] = yellow_bgr
        else:
            overlay[mask_indices] = yellow_bgr
        return overlay

    def scale_to_label(self, img, label, zoom=None):
        """
        Scale image to fill label area while maintaining aspect ratio, then apply
        the global zoom and global pan (self._pan).
        - self._pan is in image pixel coordinates (image-space offset).
        - zoom is optional; if None, use self.zoom_factor.
        """
        if zoom is None:
            zoom = self.zoom_factor
        label_width = label.width() if label.width() > 0 else 350
        label_height = label.height() if label.height() > 0 else 600
        h, w = img.shape[:2]

        # Scale (this scale is px per image pixel after fitting and zoom)
        scale = min(label_width / w, label_height / h) * zoom

        # Use global pan (image-space offsets)
        pan = getattr(self, "_pan", [0.0, 0.0])

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
            mn = float(img.min())
            mx = float(img.max())
            if mx - mn == 0:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = (255 * (img - mn) / (mx - mn)).astype(np.uint8)
        if img.ndim == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        elif img.ndim == 3 and img.shape[2] == 4:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            h, w, ch = rgba.shape
            qimg = QImage(rgba.data, w, h, ch * w, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Unsupported image shape for QPixmap conversion: {img.shape}")
        return QPixmap.fromImage(qimg.copy())

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Image Diff Viewer")
        self.resize(1200, 800)
        self.setMinimumSize(1000, 600)

        # Global pan (image-space) and zoom used by all panels
        self._pan = [0.0, 0.0]     # [x, y] image-space offset (pixels)
        self.zoom_factor = 1.0     # shared zoom

        self._init_ui()

        self.rendererA = None
        self.rendererB = None
        self.pageA = 0
        self.pageB = 0
        self.page_countA = 1
        self.page_countB = 1
        self.dpi = 150
        self.register = PageRegister()
        self.diff_engine = DiffEngine()
        self.diffs = []
        self.diff_bboxes = []
        self.current_diff_idx = 0

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
        main_layout = QVBoxLayout()

        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # File menu
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

        # View menu
        view_menu = QMenu("View", self)
        self.mode_group = QActionGroup(self)
        self.action_highlight_view = QAction("Highlight", self, checkable=True, checked=True)
        self.action_slider_view = QAction("Slider", self, checkable=True)
        self.mode_group.setExclusive(True)
        self.mode_group.addAction(self.action_highlight_view)
        self.mode_group.addAction(self.action_slider_view)
        self.action_next_diff = QAction("Next Diff", self, toolTip="Next Diff (Alt+→)")
        self.action_prev_diff = QAction("Previous Diff", self, toolTip="Previous Diff (Alt+←)")
        self.action_reset_view = QAction("Reset View", self, toolTip="Reset View (Ctrl+↓)")
        view_menu.addActions(self.mode_group.actions())
        view_menu.addSeparator()
        view_menu.addAction(self.action_next_diff)
        view_menu.addAction(self.action_prev_diff)
        view_menu.addAction(self.action_reset_view)
        view_button = QToolButton()
        view_button.setText("View")
        view_button.setMenu(view_menu)
        view_button.setPopupMode(QToolButton.InstantPopup)
        view_button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; height: 0; }")
        toolbar.addWidget(view_button)

        # Actions
        self.action_compare = QAction("Compare", self)
        self.action_prev = QAction("Previous Page", self, toolTip="Previous Page (Ctrl+←)")
        self.action_next = QAction("Next Page", self, toolTip="Next Page (Ctrl+→)")
        toolbar.addAction(self.action_compare)
        toolbar.addAction(self.action_prev)
        toolbar.addAction(self.action_next)

        # --- Highlight View Container (three-panel layout) ---
        self.highlight_view_container = QWidget()
        img_layout = QHBoxLayout(self.highlight_view_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        self.imgA_label = ZoomLabel("PDF A", self, 'A')
        self.imgA_label.setAlignment(Qt.AlignCenter)
        self.imgA_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.imgA_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.imgA_label.setMinimumSize(350, 600)
        self.imgB_label = ZoomLabel("PDF B", self, 'B')
        self.imgB_label.setAlignment(Qt.AlignCenter)
        self.imgB_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.imgB_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.imgB_label.setMinimumSize(350, 600)
        self.diff_label = ZoomLabel("Diff", self, 'D')
        self.diff_label.setAlignment(Qt.AlignCenter)
        self.diff_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.diff_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.diff_label.setMinimumSize(350, 600)
        img_layout.addWidget(self.imgA_label, stretch=1)
        img_layout.addWidget(self.imgB_label, stretch=1)
        img_layout.addWidget(self.diff_label, stretch=1)

        # --- Slider View Container (single composite widget) ---
        self.slider_view_container = QWidget()
        slider_layout = QVBoxLayout(self.slider_view_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_composite = SliderCompositeWidget(self)
        slider_layout.addWidget(self.slider_composite)

        # --- QStackedWidget to switch between highlight and slider views ---
        from PySide6.QtWidgets import QStackedWidget
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.highlight_view_container)  # index 0
        self.stacked_widget.addWidget(self.slider_view_container)     # index 1
        main_layout.addWidget(self.stacked_widget)

        # Slider for slider view (controls reveal boundary)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.slider_composite.set_slider_value)
        self.slider.hide()
        main_layout.addWidget(self.slider)

        # Bottom bar
        bottom_bar = QHBoxLayout()
        self.page_label = QLabel("Page: 1")
        self.page_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bottom_bar.addWidget(self.page_label)
        bottom_bar.addStretch(1)

        # spinner - small centered overlay; created before central so resizeEvent handles centering
        self.spinner = LoadingSpinner(self, size=48)
        self.spinner.setStyleSheet("background: rgba(255,255,255,200); border-radius: 8px;")
        self.spinner.setFixedSize(80, 80)
        self.spinner.hide()  # start hidden

        zoom_label = QLabel("Zoom:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(1000)
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

        # Ensure spinner is on top and centered initially
        self.spinner.raise_()
        self.spinner.move(self.width() // 2 - self.spinner.width() // 2,
                          self.height() // 2 - self.spinner.height() // 2)

        # Connections
        self.action_loadA.triggered.connect(self.load_pdf_a)
        self.action_loadB.triggered.connect(self.load_pdf_b)
        self.action_export.triggered.connect(self.export_compared_pdf)
        self.action_compare.triggered.connect(self.compare)
        self.action_prev.triggered.connect(self.prev_page)
        self.action_next.triggered.connect(self.next_page)
        self.action_next_diff.triggered.connect(lambda: self.goto_diff(1))
        self.action_prev_diff.triggered.connect(lambda: self.goto_diff(-1))
        self.action_reset_view.triggered.connect(self.reset_view)
        self.action_highlight_view.triggered.connect(lambda: self.change_view_mode("highlight"))
        self.action_slider_view.triggered.connect(lambda: self.change_view_mode("slider"))

        QShortcut(QKeySequence("Ctrl+Right"), self, self.next_page)
        QShortcut(QKeySequence("Ctrl+Left"), self, self.prev_page)
        QShortcut(QKeySequence("Ctrl+Down"), self, self.reset_view)
        QShortcut(QKeySequence("Alt+Right"), self, lambda: self.goto_diff(1))
        QShortcut(QKeySequence("Alt+Left"), self, lambda: self.goto_diff(-1))

    def change_view_mode(self, mode):
        self.current_mode = mode
        if mode == "highlight":
            self.stacked_widget.setCurrentIndex(0)
            self.slider.hide()
        elif mode == "slider":
            self.stacked_widget.setCurrentIndex(1)
            self.slider.show()
        self.slider_composite.repaint()

    def export_compared_pdf(self):
        if not self.diffs:
            QMessageBox.warning(self, "No Diff", "Please compare PDFs first.")
            return
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Save Compared PDF", "", "PDF Files (*.pdf)")
            if not path:
                return
            import fitz  # PyMuPDF
            doc = fitz.open()
            for img in self.diffs:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("Unsupported image shape for export")
                height, width = img.shape[:2]
                success, png_bytes = cv2.imencode('.png', img)
                if not success:
                    raise RuntimeError("Failed to encode image as PNG for PDF export")
                png_bytes = png_bytes.tobytes()
                page = doc.new_page(width=width, height=height)
                page.insert_image(page.rect, stream=png_bytes)
            doc.save(path)
            doc.close()
            QMessageBox.information(self, "Export Complete", f"Compared PDF saved to:\n{path}")
        except Exception as e:
            logging.error(f"Failed to export compared PDF: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export compared PDF: {e}")

    def update_page_label(self):
        max_pages = max(getattr(self, 'page_countA', 1), getattr(self, 'page_countB', 1))
        self.page_label.setText(f"Page: {getattr(self, 'pageA', 0) + 1} / {max_pages}")
        if hasattr(self, 'action_export'):
            self.action_export.setEnabled(bool(self.diffs))

    def _reset_view_state(self):
        # Reset global pan and zoom
        self._pan = [0.0, 0.0]
        self.zoom_factor = 1.0
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(self.zoom_factor * 100))
        self.zoom_slider.blockSignals(False)
        self.zoom_input.blockSignals(True)
        self.zoom_input.setText(f"{self.zoom_factor:.2f}")
        self.zoom_input.blockSignals(False)

    def reset_view(self):
        self._reset_view_state()
        self.show_page_a()
        self.show_page_b()
        self.show_diff()
        # Also reset and update slider view
        if hasattr(self, 'slider_composite'):
            self.slider_composite.repaint()

    def load_pdf_a(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open PDF A", "", "PDF Files (*.pdf)")
            if path:
                self.rendererA = PDFRenderer(path)
                self.pageA = 0
                self.page_countA = self.rendererA.get_page_count()
                self._reset_view_state()
                self.show_page_a()
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
                img = self.scale_to_label(img, self.imgA_label, zoom=self.zoom_factor)
                pixmap = self.np_to_pixmap(img)
                self.imgA_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Failed to render PDF A page: {e}")
            QMessageBox.critical(self, "Error", f"Failed to render PDF A page: {e}")

    def show_page_b(self):
        try:
            if self.rendererB:
                img = self.rendererB.render_page(self.pageB, dpi=self.dpi)
                img = self.scale_to_label(img, self.imgB_label, zoom=self.zoom_factor)
                pixmap = self.np_to_pixmap(img)
                self.imgB_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Failed to render PDF B page: {e}")
            QMessageBox.critical(self, "Error", f"Failed to render PDF B page: {e}")

    def prev_page(self):
        if self.rendererA and self.pageA > 0:
            self.pageA -= 1
        if self.rendererB and self.pageB > 0:
            self.pageB -= 1
        self.current_diff_idx = 0
        self.show_page_a()
        self.show_page_b()
        self.show_diff()
        self.update_page_label()
        # Also update slider view
        if hasattr(self, 'slider_composite'):
            self.slider_composite.repaint()

    def next_page(self):
        if self.rendererA and self.pageA < self.page_countA - 1:
            self.pageA += 1
        if self.rendererB and self.pageB < self.page_countB - 1:
            self.pageB += 1
        self.current_diff_idx = 0
        self.show_page_a()
        self.show_page_b()
        self.show_diff()
        self.update_page_label()
        # Also update slider view
        if hasattr(self, 'slider_composite'):
            self.slider_composite.repaint()

    def compare(self):
        if not (self.rendererA and self.rendererB):
            return

        self.spinner.start()

        from src.worker_compare import CompareWorker
        self.thread = QThread()
        self.worker = CompareWorker(self.rendererA, self.rendererB,
                                    self.dpi, self.register,
                                    self.diff_engine)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._compare_done)
        self.worker.error.connect(self._compare_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _compare_done(self, diffs, bboxes):
        self.spinner.stop()
        self.diffs = diffs
        self.diff_bboxes = bboxes
        self.current_diff_idx = 0
        self.show_diff()
        self.action_export.setEnabled(True)

    def _compare_error(self, msg):
        self.spinner.stop()
        QMessageBox.critical(self, "Error", msg)

    def goto_diff(self, direction=1):
        if not (self.diffs and self.pageA < len(self.diff_bboxes)):
            return
        bboxes = self.diff_bboxes[self.pageA]
        if not bboxes:
            return
        
        self.current_diff_idx = (self.current_diff_idx + direction) % len(bboxes)
        x, y, w_box, h_box = bboxes[self.current_diff_idx]

        margin_x = max(int(w_box * 0.05), 5)
        margin_y = max(int(h_box * 0.05), 5)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = x + w_box + margin_x
        y1 = y + h_box + margin_y
        bbox_w = x1 - x0
        bbox_h = y1 - y0

        diff_img = self.diffs[self.pageA]
        img_h, img_w = diff_img.shape[:2]

        label = self.imgA_label
        label_width = label.width() if label.width() > 0 else 350
        label_height = label.height() if label.height() > 0 else 600

        base_scale_x = label_width / img_w
        base_scale_y = label_height / img_h
        base_scale = min(base_scale_x, base_scale_y)

        req_scale_x = label_width / bbox_w
        req_scale_y = label_height / bbox_h
        req_final_scale = min(req_scale_x, req_scale_y)

        if base_scale <= 0:
            target_zoom = 1.0
        else:
            target_zoom = req_final_scale / base_scale

        target_zoom = max(0.1, min(target_zoom, 10.0))

        cx_img = x0 + bbox_w / 2.0
        cy_img = y0 + bbox_h / 2.0

        pan_x = (img_w / 2.0) - cx_img
        pan_y = (img_h / 2.0) - cy_img

        # set global pan
        self._pan = [pan_x, pan_y]

        # apply zoom and redraw
        self.set_zoom(target_zoom, center=None, update_controls=True)
        # For slider mode: set margin-adjusted temp rect and repaint after pan/zoom update
        if hasattr(self, 'slider_temp_diff_rect'):
            self.slider_temp_diff_rect = (x0, y0, bbox_w, bbox_h)
        if hasattr(self, 'slider_composite'):
            self.slider_composite.repaint()

        try:
            disp = diff_img.copy()
            x0i = int(x0); y0i = int(y0); x1i = int(x1); y1i = int(y1)
            cv2.rectangle(disp, (x0i, y0i), (x1i, y1i), (0, 0, 255), 2)
            disp_scaled = self.scale_to_label(disp, self.diff_label, zoom=self.zoom_factor)
            pixmap = self.np_to_pixmap(disp_scaled)
            self.diff_label.setPixmap(pixmap)
        except Exception:
            pass

    def show_diff(self):
        if self.diffs and 0 <= self.pageA < len(self.diffs):
            diff_img = self.diffs[self.pageA]
            diff_img = self.scale_to_label(diff_img, self.diff_label, zoom=self.zoom_factor)
            pixmap = self.np_to_pixmap(diff_img)
            self.diff_label.setPixmap(pixmap)
        else:
            self.diff_label.clear()

    def set_zoom(self, factor, center=None, update_controls=True):
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(factor, 10.0))
        # If a center is provided (label-local coords), adjust global pan so that
        # the point under the cursor stays stable.
        if center is not None:
            # center is expected to be tuple (label, (x,y)) or None.
            # For backward compatibility accept (x,y) as label-local coords on active label.
            if (
                isinstance(center, tuple)
                and len(center) == 2
                and isinstance(center[0], QWidget)
            ):
                label, (cx, cy) = center
                # Convert center from label-local to image-space and update self._pan
                self._adjust_global_pan_from_label_center(label, (cx, cy), old_zoom, self.zoom_factor)
            elif isinstance(center, tuple) and len(center) == 2:
                # No label given, assume center coordinates refers to active label (diff_label)
                self._adjust_global_pan_from_label_center(self.diff_label, center, old_zoom, self.zoom_factor)
        if update_controls:
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(int(self.zoom_factor * 100))
            self.zoom_slider.blockSignals(False)
            self.zoom_input.blockSignals(True)
            self.zoom_input.setText(f"{self.zoom_factor:.2f}")
            self.zoom_input.blockSignals(False)
        self.show_page_a()
        self.show_page_b()
        self.show_diff()

    def _adjust_global_pan_from_label_center(self, label, center, old_zoom, new_zoom):
        """
        Given a label and a center point in label-local coords, compute the global image-space pan
        so that the image point under that center remains under the center after zoom change.
        """
        if center is None:
            return
        label_width = label.width() if label.width() > 0 else 350
        label_height = label.height() if label.height() > 0 else 600
        x, y = center
        # compute image-space point under (x,y) at old zoom
        img_x = (x - label_width / 2.0) / old_zoom - self._pan[0]
        img_y = (y - label_height / 2.0) / old_zoom - self._pan[1]
        # compute new pan so image point remains under same label coords
        new_pan_x = (x - label_width / 2.0) / new_zoom - img_x
        new_pan_y = (y - label_height / 2.0) / new_zoom - img_y
        self._pan = [new_pan_x, new_pan_y]

    def _on_zoom_slider(self, value):
        zoom = value / 100.0
        self.set_zoom(zoom, center=None, update_controls=True)

    def _on_zoom_input(self):
        try:
            zoom = float(self.zoom_input.text())
        except Exception:
            zoom = self.zoom_factor
        self.set_zoom(zoom, center=None, update_controls=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "spinner"):
            w = self.width()
            h = self.height()
            self.spinner.move(w // 2 - self.spinner.width() // 2,
                              h // 2 - self.spinner.height() // 2)
            
            

class SliderCompositeWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.slider_value = 50  # 0 = all A, 100 = all B
        self.setMinimumSize(350, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._drag_active = False
        self._last_pos = None
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        viewer = self.parent
        zoom_step = 1.25
        old_zoom = viewer.zoom_factor
        if delta > 0:
            new_zoom = min(old_zoom * zoom_step, 10.0)
        else:
            new_zoom = max(old_zoom / zoom_step, 0.1)
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        # Use mouse position as zoom center
        pos = event.position() if hasattr(event, 'position') else event.posF()
        cx, cy = int(pos.x()), int(pos.y())
        viewer.set_zoom(new_zoom, center=(self, (cx, cy)), update_controls=True)
        self.repaint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._last_pos = event.pos()
            # Clear temp diff rect on user interaction
            if hasattr(self.parent, 'slider_temp_diff_rect') and self.parent.slider_temp_diff_rect is not None:
                self.parent.slider_temp_diff_rect = None
                self.repaint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        viewer = self.parent
        if (event.buttons() & Qt.LeftButton) and self._drag_active and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            speed = 2.0
            dx = int(delta.x() * speed / max(viewer.zoom_factor, 1e-6))
            dy = int(delta.y() * speed / max(viewer.zoom_factor, 1e-6))
            viewer._pan[0] += dx
            viewer._pan[1] += dy
            viewer.show_page_a()
            viewer.show_page_b()
            viewer.show_diff()
            self.repaint()
            # Clear temp diff rect on user drag
            if hasattr(viewer, 'slider_temp_diff_rect') and viewer.slider_temp_diff_rect is not None:
                viewer.slider_temp_diff_rect = None
                self.repaint()
        else:
            self._drag_active = False
            self._last_pos = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            self._last_pos = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._drag_active = False
        self._last_pos = None
        super().leaveEvent(event)

    def set_slider_value(self, value):
        self.slider_value = value
        self.repaint()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor
        viewer = self.parent
        if not (viewer.rendererA and viewer.rendererB):
            return
        imgA = viewer.rendererA.render_page(viewer.pageA, dpi=viewer.dpi)
        imgB = viewer.rendererB.render_page(viewer.pageB, dpi=viewer.dpi)
        # Apply zoom/pan and scale to widget size
        imgA = viewer.scale_to_label(imgA, self, zoom=viewer.zoom_factor)
        imgB = viewer.scale_to_label(imgB, self, zoom=viewer.zoom_factor)
        # Convert to QImage
        qimgA = viewer.np_to_pixmap(imgA).toImage()
        qimgB = viewer.np_to_pixmap(imgB).toImage()
        painter = QPainter(self)
        # Draw PDF A fully
        painter.drawImage(0, 0, qimgA)
        # Clip and draw PDF B according to slider
        w = self.width()
        h = self.height()
        reveal_x = int(w * self.slider_value / 100)
        if reveal_x < w:
            painter.save()
            painter.setClipRect(reveal_x, 0, w - reveal_x, h)
            painter.drawImage(0, 0, qimgB)
            painter.restore()
        # Draw diff rectangles only in highlight mode
        if hasattr(viewer, 'current_mode') and viewer.current_mode == "highlight":
            if viewer.diff_bboxes and viewer.pageA < len(viewer.diff_bboxes):
                bboxes = viewer.diff_bboxes[viewer.pageA]
                for idx, (x, y, bw, bh) in enumerate(bboxes):
                    # Transform bbox to widget coordinates
                    img_h, img_w = imgA.shape[:2]
                    scale_x = w / img_w
                    scale_y = h / img_h
                    rect_x = int(x * scale_x)
                    rect_y = int(y * scale_y)
                    rect_w = int(bw * scale_x)
                    rect_h = int(bh * scale_y)
                    pen = QPen(QColor(255, 0, 0), 2)
                    painter.setPen(pen)
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
        # Draw temp diff rectangle overlay in slider mode if set
        rect = getattr(viewer, 'slider_temp_diff_rect', None)
        if rect is not None and hasattr(viewer, 'current_mode') and viewer.current_mode == "slider":
            x, y, bw, bh = rect
            img_h, img_w = imgA.shape[:2]
            scale_x = w / img_w
            scale_y = h / img_h
            rect_x = int(x * scale_x)
            rect_y = int(y * scale_y)
            rect_w = int(bw * scale_x)
            rect_h = int(bh * scale_y)
            pen = QPen(QColor(255, 0, 0), 3)
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)
        painter.end()


# Custom QLabel to handle wheel events for zoom & drag panning
class ZoomLabel(QLabel):
    def __init__(self, text, parent, which):
        super().__init__(text)
        self.parent = parent
        self.which = which  # 'A', 'B', or 'D'
        self.setMouseTracking(True)
        self._drag_active = False
        self._last_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_step = 1.25
        old_zoom = self.parent.zoom_factor
        if delta > 0:
            new_zoom = min(old_zoom * zoom_step, 10.0)
        else:
            new_zoom = max(old_zoom / zoom_step, 0.1)

        if abs(new_zoom - old_zoom) < 1e-6:
            return

        # Update global zoom and for each panel adjust global pan to keep the relevant point stable.
        # We will prioritize the actual panel under the cursor (self) for pan stability by using its center.
        self.parent.zoom_factor = new_zoom
        # If the cursor is over this specific label, use its local coords as center; otherwise use panel center.

        # Update controls
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
        if event.button() == Qt.LeftButton and self.which in ['A', 'B']:
            if self.which == 'A' and not self.parent.rendererA:
                self.parent.load_pdf_a()
            elif self.which == 'B' and not self.parent.rendererB:
                self.parent.load_pdf_b()

        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._last_pos = event.pos()

    def mouseMoveEvent(self, event):
        # Only drag if left button is currently pressed
        if (event.buttons() & Qt.LeftButton) and self._drag_active and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            speed = 2.0
            dx = int(delta.x() * speed / max(self.parent.zoom_factor, 1e-6))
            dy = int(delta.y() * speed / max(self.parent.zoom_factor, 1e-6))
            self.parent._pan[0] += dx
            self.parent._pan[1] += dy
            self.parent.show_page_a()
            self.parent.show_page_b()
            self.parent.show_diff()
        else:
            # If the left button is not pressed, stop dragging
            self._drag_active = False
            self._last_pos = None
        super().mouseMoveEvent(event)
    def leaveEvent(self, event):
        # If the mouse leaves the widget, stop dragging
        self._drag_active = False
        self._last_pos = None
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            self._last_pos = None

    def adjust_pan_for_zoom(self, center, old_zoom, new_zoom):
        """
        Backwards-compatible stub: delegate to parent helper with this label as context.
        center: (x, y) label-local coords
        """
        if center is None:
            return
        self.parent._adjust_global_pan_from_label_center(self, center, old_zoom, new_zoom)

    @staticmethod
    def overlay_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
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

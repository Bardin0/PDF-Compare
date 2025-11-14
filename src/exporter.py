# src/exporter.py
"""
PDF/image export logic using img2pdf and ReportLab.
"""

import img2pdf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from typing import List
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.INFO)

class Exporter:
    @staticmethod
    def export_images_to_pdf(image_paths: List[str], output_pdf: str):
        try:
            with open(output_pdf, "wb") as f:
                f.write(img2pdf.convert(image_paths))
        except Exception as e:
            logging.error(f"Failed to export images to PDF '{output_pdf}': {e}")

    @staticmethod
    def export_canvas(images: List[Image.Image], output_pdf: str):
        try:
            c = canvas.Canvas(output_pdf, pagesize=letter)
            for img in images:
                img_path = "_temp_export_img.png"
                img.save(img_path)
                c.drawImage(img_path, 0, 0, width=letter[0], height=letter[1])
                c.showPage()
                os.remove(img_path)
            c.save()
        except Exception as e:
            logging.error(f"Failed to export canvas to PDF '{output_pdf}': {e}")

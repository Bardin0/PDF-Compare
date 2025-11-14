import unittest
import os
from src.renderer import PDFRenderer

class TestPDFRenderer(unittest.TestCase):
    def setUp(self):
        self.sample_pdf = "tests/sample.pdf"  # Place a small sample PDF here for CI
        if not os.path.exists(self.sample_pdf):
            self.skipTest("Sample PDF not found.")
        self.renderer = PDFRenderer(self.sample_pdf)

    def tearDown(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()

    def test_page_count(self):
        count = self.renderer.get_page_count()
        self.assertGreater(count, 0)

    def test_render_page(self):
        img = self.renderer.render_page(0, dpi=72)
        self.assertIsNotNone(img)
        self.assertEqual(img.ndim, 3)

if __name__ == "__main__":
    unittest.main()

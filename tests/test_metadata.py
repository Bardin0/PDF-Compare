import unittest
from src.metadata import MetadataExtractor
import os

class TestMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.sample_pdf = "tests/sample.pdf"
        if not os.path.exists(self.sample_pdf):
            self.skipTest("Sample PDF not found.")
        self.extractor = MetadataExtractor(self.sample_pdf)

    def tearDown(self):
        if hasattr(self, 'extractor'):
            self.extractor.close()

    def test_extract_text_blocks(self):
        blocks = self.extractor.extract_text_blocks(0)
        self.assertIsInstance(blocks, dict)

if __name__ == "__main__":
    unittest.main()

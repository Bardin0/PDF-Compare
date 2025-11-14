import unittest
import numpy as np
from src.diff_engine import DiffEngine

class TestDiffEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DiffEngine()
        self.imgA = np.zeros((100, 100, 3), dtype=np.uint8)
        self.imgB = np.zeros((100, 100, 3), dtype=np.uint8)
        self.imgB[50:60, 50:60] = 255

    def test_diff(self):
        mask = self.engine.compute_diff(self.imgA, self.imgB)
        self.assertEqual(mask.shape, self.imgA.shape[:2])
        self.assertTrue(np.any(mask > 0))

if __name__ == "__main__":
    unittest.main()

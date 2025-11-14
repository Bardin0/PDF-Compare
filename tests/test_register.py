import unittest
import numpy as np
from src.register import PageRegister

class TestPageRegister(unittest.TestCase):
    def setUp(self):
        self.reg = PageRegister()
        # Use a white square on black for features
        self.imgA = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2 = __import__('cv2')
        cv2.rectangle(self.imgA, (30, 30), (70, 70), (255, 255, 255), -1)
        self.imgB = self.imgA.copy()

    def test_align_identity(self):
        aligned, H = self.reg.align(self.imgA, self.imgA)
        self.assertIsNotNone(H)
        self.assertEqual(aligned.shape, self.imgA.shape)

    def test_align_shift(self):
        aligned, H = self.reg.align(self.imgA, self.imgB)
        self.assertEqual(aligned.shape, self.imgA.shape)

if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.preprocessing import TextPreprocessor, WordSegmenter  # noqa: E402


class PreprocessingTest(unittest.TestCase):
    def test_clean_normalize_emoji_and_abbreviation(self) -> None:
        processor = TextPreprocessor(
            use_word_segmentation=True,
            segmenter=WordSegmenter(backend="none"),
        )
        text = '<b>sp</b> ok :) https://example.com giá tốt <3'
        processed = processor.transform(text)
        self.assertIn("sản_phẩm", processed)
        self.assertIn("vui_vẻ", processed)
        self.assertIn("yêu_thích", processed)
        self.assertNotIn("http", processed)
        self.assertNotIn("<b>", processed)

    def test_vocabulary_standardization(self) -> None:
        processor = TextPreprocessor(use_word_segmentation=False)
        processed = processor.transform("ko đc liên hệ cskh")
        self.assertIn("không", processed)
        self.assertIn("được", processed)
        self.assertIn("chăm sóc khách hàng", processed)


if __name__ == "__main__":
    unittest.main()

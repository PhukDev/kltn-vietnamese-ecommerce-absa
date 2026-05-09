from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.absa import normalize_aspect_label, prepare_absa_annotation_frame  # noqa: E402


class AbsaAnnotationTest(unittest.TestCase):
    def test_normalize_aspect_label(self) -> None:
        self.assertEqual(normalize_aspect_label("1"), 1)
        self.assertEqual(normalize_aspect_label("0"), 0)
        self.assertEqual(normalize_aspect_label("x"), 1)
        self.assertIsNone(normalize_aspect_label(""))
        self.assertIsNone(normalize_aspect_label("maybe"))

    def test_prepare_absa_annotation_frame_skips_unlabeled_rows(self) -> None:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise unittest.SkipTest("pandas is not installed") from exc

        frame = pd.DataFrame(
            [
                {
                    "content": "dịch vụ hỗ trợ tốt",
                    "product": "0",
                    "price": "0",
                    "delivery": "0",
                    "service": "1",
                    "app": "0",
                },
                {
                    "content": "chưa gán nhãn",
                    "product": "",
                    "price": "",
                    "delivery": "",
                    "service": "",
                    "app": "",
                },
            ]
        )
        prepared = prepare_absa_annotation_frame(frame)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(int(prepared.iloc[0]["service"]), 1)
        self.assertIn("processed_text", prepared.columns)


if __name__ == "__main__":
    unittest.main()

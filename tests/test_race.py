from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.race import analyze_business_signal, detect_aspects  # noqa: E402


class RaceTest(unittest.TestCase):
    def test_detect_service_aspect(self) -> None:
        self.assertIn("service", detect_aspects("dịch vụ chăm sóc khách hàng quá tệ"))

    def test_negative_service_high_thumbsup_alert(self) -> None:
        signal = analyze_business_signal(
            content="dịch vụ hỗ trợ quá tệ",
            sentiment="negative",
            thumbsupcount=20,
        )
        self.assertEqual(signal["race"]["primary_stage"], "Engage")
        self.assertEqual(signal["alert"]["severity"], "high")


if __name__ == "__main__":
    unittest.main()

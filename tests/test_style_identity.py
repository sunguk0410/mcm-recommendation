import unittest
from types import SimpleNamespace

from src.style_identity import (
    build_fallback_title,
    generate_style_identity_title,
)


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


class StyleIdentityTest(unittest.TestCase):
    def setUp(self):
        self.products = [{
            "productId": 70,
            "name": "Ottomar 위켄더",
            "category": "BAG",
            "subCategory": None,
            "color": "Black",
            "zone": "TRAVEL",
            "directInterestScore": 7.5,
            "evidence": ["ONLINE_WISHLIST", "FITTING"],
        }]

    def test_returns_valid_generated_korean_noun_phrase(self):
        client = FakeClient("자유로운 도시의 노마드")

        title = generate_style_identity_title(
            self.products,
            [{"category": "BAG", "zone": "TRAVEL", "dwellSeconds": 375}],
            client=client,
            model="test-model",
        )

        self.assertEqual("자유로운 도시의 노마드", title)
        self.assertEqual("test-model", client.responses.request["model"])
        self.assertIn("Ottomar 위켄더", client.responses.request["input"])

    def test_invalid_response_uses_dominant_zone_fallback(self):
        client = FakeClient("고객은 여행을 좋아하는 사람입니다.")

        title = generate_style_identity_title(
            self.products,
            [{"category": "BAG", "zone": "TRAVEL", "dwellSeconds": 375}],
            client=client,
        )

        self.assertEqual("자유로운 여정의 탐험가", title)

    def test_fallback_uses_highest_dwell_zone(self):
        title = build_fallback_title(
            self.products,
            [
                {"category": "BAG", "zone": "NEW", "dwellSeconds": 55},
                {"category": "BAG", "zone": "CLASSIC", "dwellSeconds": 115},
                {"category": "BAG", "zone": "TRAVEL", "dwellSeconds": 375},
            ],
        )

        self.assertEqual("자유로운 여정의 탐험가", title)


if __name__ == "__main__":
    unittest.main()

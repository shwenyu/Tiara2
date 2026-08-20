import json
import tempfile
import unittest
from pathlib import Path

from tiara.config import load_config


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        config = load_config()
        self.assertEqual(config["runtime"]["min_length"], 1000)
        self.assertEqual(config["router"], {})

    def test_documented_config(self):
        config = load_config(Path(__file__).parents[1] / "config" / "default.json")
        self.assertEqual(config["router"]["length_bins_bp"], [2500, 5000])
        self.assertFalse(config["router"]["demotion_allowed"])

    def test_rejects_unknown_router_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"router": {"unknown": 1}}))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

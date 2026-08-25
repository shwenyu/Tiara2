import hashlib
import tempfile
import unittest
from pathlib import Path

from tiara.bundle import _download


class BundleDownloadTest(unittest.TestCase):
    def test_download_verifies_sha256(self):
        payload = b"small deterministic model fixture"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "cache" / "model.bin"
            source.write_bytes(payload)
            result = _download(source.as_uri(), destination, expected, len(payload))
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(result["sha256"], expected)

    def test_download_rejects_wrong_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "model.bin"
            source.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                _download(source.as_uri(), destination, "0" * 64)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("model.bin.part").exists())


if __name__ == "__main__":
    unittest.main()

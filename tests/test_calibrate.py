import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tiara.calibrate import calibrate
from tiara.cli import parser
from tiara.hierarchical.biosignal_multi_expert import apply_eukaryote_threshold


class EukThresholdTest(unittest.TestCase):
    def test_omitted_threshold_preserves_root_argmax(self):
        probabilities = np.asarray([[0.45, 0.40, 0.15], [0.20, 0.70, 0.10]])
        choices = apply_eukaryote_threshold(
            probabilities, ["euk_nuclear", "prok", "organelle"], None
        )
        np.testing.assert_array_equal(choices, [0, 1])

    def test_threshold_can_demote_or_promote_eukaryote(self):
        probabilities = np.asarray([[0.45, 0.40, 0.15], [0.41, 0.42, 0.17]])
        demoted = apply_eukaryote_threshold(
            probabilities[:1], ["euk_nuclear", "prok", "organelle"], 0.50
        )
        promoted = apply_eukaryote_threshold(
            probabilities[1:], ["euk_nuclear", "prok", "organelle"], 0.40
        )
        np.testing.assert_array_equal(demoted, [1])
        np.testing.assert_array_equal(promoted, [0])

    def test_cli_accepts_one_optional_threshold(self):
        args = parser().parse_args(["-i", "input.fa", "-o", "output.tsv", "--euk-threshold", "0.5"])
        self.assertEqual(args.euk_threshold, 0.5)

    def test_cli_rejects_out_of_range_threshold(self):
        with self.assertRaises(SystemExit):
            parser().parse_args(["-i", "input.fa", "-o", "output.tsv", "--euk-threshold", "1.1"])

    def test_optional_validation_calibration_maximizes_observed_f1(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            predictions = directory / "predictions.tsv"
            truth = directory / "truth.tsv"
            with predictions.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("record_id", "p_eukaryote"))
                writer.writerows((("p1", 0.9), ("p2", 0.6), ("n1", 0.55), ("n2", 0.1)))
            with truth.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("record_id", "truth_class"))
                writer.writerows((("p1", "eukaryote"), ("p2", "eukaryote"), ("n1", "prokaryote"), ("n2", "organelle")))
            result = calibrate(predictions, truth)
            self.assertEqual(result["recommended_euk_threshold"], 0.6)
            self.assertEqual(result["metrics_at_recommended_threshold"]["f1"], 1.0)
            self.assertEqual(result["objective"], "maximum_observed_f1")

    def test_calibration_rejects_empty_and_invalid_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            predictions = directory / "predictions.tsv"
            truth = directory / "truth.tsv"
            predictions.write_text("record_id\tp_eukaryote\n", encoding="utf-8")
            truth.write_text("record_id\ttruth_class\np1\teukaryote\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contains no records"):
                calibrate(predictions, truth)

            predictions.write_text(
                "record_id\tp_eukaryote\np1\t1.2\nn1\t0.1\n", encoding="utf-8"
            )
            truth.write_text(
                "record_id\ttruth_class\np1\teukaryote\nn1\tprokaryote\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "finite and in"):
                calibrate(predictions, truth)


if __name__ == "__main__":
    unittest.main()

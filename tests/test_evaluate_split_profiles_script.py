from __future__ import annotations

import unittest

from scripts.evaluate_split_profiles import (
    _build_split_manifest_payload,
    _counts,
    _split_ids,
    _split_ids_grouped,
    _validate_split_ids,
)


class EvaluateSplitProfilesScriptTests(unittest.TestCase):
    def test_split_ids_disjoint_and_complete(self) -> None:
        labels = {
            1: True,
            2: True,
            3: False,
            4: False,
            5: False,
            6: False,
            7: True,
            8: False,
            9: False,
            10: False,
        }
        hardness = {job_id: float(job_id) / 10.0 for job_id in labels}
        split_ids = _split_ids(
            job_ids=sorted(labels.keys()),
            labels=labels,
            hardness_by_id=hardness,
            dev_ratio=0.6,
            test_ratio=0.2,
            challenge_ratio=0.2,
            seed=42,
        )
        dev = set(split_ids["dev"])
        test = set(split_ids["test"])
        challenge = set(split_ids["challenge"])
        self.assertFalse(dev & test)
        self.assertFalse(dev & challenge)
        self.assertFalse(test & challenge)
        self.assertEqual(dev | test | challenge, set(labels.keys()))

    def test_counts(self) -> None:
        labels = {1: True, 2: False, 3: False, 4: True}
        c = _counts(labels, [1, 2, 3])
        self.assertEqual(c["jobs"], 3)
        self.assertEqual(c["critical_positives"], 1)
        self.assertEqual(c["critical_negatives"], 2)

    def test_grouped_split_preserves_group_boundary(self) -> None:
        labels = {
            1: True,
            2: False,
            3: True,
            4: False,
            5: False,
            6: False,
            7: True,
            8: False,
        }
        hardness = {job_id: float(job_id) / 10.0 for job_id in labels}
        group_by_id = {
            1: "g1",
            2: "g1",
            3: "g2",
            4: "g2",
            5: "g3",
            6: "g3",
            7: "g4",
            8: "g4",
        }
        split_ids = _split_ids_grouped(
            job_ids=sorted(labels.keys()),
            labels=labels,
            hardness_by_id=hardness,
            group_by_id=group_by_id,
            dev_ratio=0.6,
            test_ratio=0.2,
            challenge_ratio=0.2,
            seed=7,
            challenge_fixed_ids=None,
        )
        split_of: dict[int, str] = {}
        for split_name, rows in split_ids.items():
            for job_id in rows:
                split_of[job_id] = split_name

        for g in {"g1", "g2", "g3", "g4"}:
            ids = [job_id for job_id, group in group_by_id.items() if group == g]
            self.assertEqual(len({split_of[job_id] for job_id in ids}), 1)

    def test_validate_split_ids_requires_full_cover(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_split_ids(
                split_ids={"dev": [1], "test": [2], "challenge": []},
                eligible_ids=[1, 2, 3],
            )

    def test_split_manifest_contains_hash(self) -> None:
        payload = _build_split_manifest_payload(
            task_id="task-a",
            strategy="group_trainee",
            holdout_site=None,
            seed=1,
            split_ratios={"dev": 0.6, "test": 0.2, "challenge": 0.2},
            split_ids={"dev": [1], "test": [2], "challenge": [3]},
        )
        self.assertIn("artifact_hash_sha256", payload)


if __name__ == "__main__":
    unittest.main()

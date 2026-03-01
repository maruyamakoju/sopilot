from __future__ import annotations

import unittest

from sopilot.eval.integrity import attach_payload_hash, payload_sha256, verify_payload_hash


class EvalIntegrityTests(unittest.TestCase):
    def test_hash_roundtrip(self) -> None:
        payload = {
            "a": 1,
            "b": {"x": 2, "y": [1, 2, 3]},
        }
        hashed = attach_payload_hash(payload)
        self.assertTrue(verify_payload_hash(hashed))

    def test_hash_excluding_policy_id(self) -> None:
        payload = {
            "version": "critical_policy_v1",
            "policy_id": "policy-foo",
            "guardrails": {"guarded_binary_v2": {"min_dtw": 0.025}},
        }
        hashed = attach_payload_hash(
            payload,
            exclude_extra_keys={"policy_id"},
            method="sha256(canonical_json,exclude=artifact_hash_sha256|artifact_hash_method|policy_id)",
        )
        self.assertTrue(verify_payload_hash(hashed, exclude_extra_keys={"policy_id"}))
        tampered = dict(hashed)
        tampered["guardrails"] = {"guarded_binary_v2": {"min_dtw": 0.03}}
        self.assertFalse(verify_payload_hash(tampered, exclude_extra_keys={"policy_id"}))

    def test_payload_sha256_stable(self) -> None:
        p1 = {"k2": 2, "k1": 1}
        p2 = {"k1": 1, "k2": 2}
        self.assertEqual(payload_sha256(p1), payload_sha256(p2))


if __name__ == "__main__":
    unittest.main()

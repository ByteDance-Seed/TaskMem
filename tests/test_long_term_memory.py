"""Regression tests for LongTermMemory.to_string().

Run from the repository root with:
    python -m unittest discover -s tests -v
"""

import unittest

from memory.long_term_memory import LongTermMemory


class LongTermMemoryToStringTests(unittest.TestCase):
    def setUp(self):
        self.memory = LongTermMemory(interval=10)

    def test_episodic_serialization_uses_descriptions(self):
        self.memory.episodic.description.update({
            "0": "EP-0 description",
            "1": "EP-1 description",
        })
        self.memory.semantic.knowledge.update({
            "0": ["SEM-0 knowledge"],
            "1": ["SEM-1 knowledge"],
        })

        result = self.memory.to_string(memory_type="episodic")

        self.assertIn("EP-0 description", result)
        self.assertIn("EP-1 description", result)
        self.assertNotIn("SEM-", result)
        self.assertIn("00:00:00-00:00:10", result)
        self.assertIn("\n\n00:00:10-00:00:20", result)

    def test_semantic_serialization_excludes_episodic_content(self):
        self.memory.episodic.description["0"] = "EP-0 description"
        self.memory.semantic.knowledge["0"] = ["SEM-0 knowledge"]

        result = self.memory.to_string(memory_type="semantic")

        self.assertIn("Knowledge:", result)
        self.assertIn("SEM-0 knowledge", result)
        self.assertNotIn("EP-0 description", result)

    def test_both_uses_union_of_keys_and_numeric_order(self):
        self.memory.episodic.description.update({
            "10": "EP-10 description",
            "2": "EP-2 description",
        })
        self.memory.semantic.knowledge.update({
            "10": ["SEM-10 knowledge"],
            "3": ["SEM-3 knowledge"],
        })

        result = self.memory.to_string(memory_type="both")

        self.assertIn("EP-2 description", result)
        self.assertIn("EP-10 description", result)
        self.assertIn("SEM-3 knowledge", result)
        self.assertIn("SEM-10 knowledge", result)
        self.assertLess(result.index("EP-2 description"), result.index("EP-10 description"))
        self.assertLess(result.index("SEM-3 knowledge"), result.index("SEM-10 knowledge"))

    def test_episodic_only_memory_has_default_output(self):
        self.memory.episodic.description["0"] = "EP-only description"

        result = self.memory.to_string(memory_type="episodic")

        self.assertIn("EP-only description", result)

    def test_semantic_only_memory_has_default_output(self):
        self.memory.semantic.knowledge["0"] = ["SEM-only knowledge"]

        result = self.memory.to_string(memory_type="semantic")

        self.assertIn("SEM-only knowledge", result)

    def test_explicit_clip_order_is_preserved(self):
        for clip_id in ("2", "5", "9"):
            self.memory.episodic.description[clip_id] = f"EP-{clip_id}"

        result = self.memory.to_string(
            memory_type="episodic",
            clip_ids=["5", "2", "9"],
        )

        self.assertLess(result.index("EP-5"), result.index("EP-2"))
        self.assertLess(result.index("EP-2"), result.index("EP-9"))

    def test_explicit_asymmetric_clip_ids_skip_missing_modalities(self):
        self.memory.episodic.description["2"] = "EP-2"
        self.memory.episodic.description["9"] = "EP-9"
        self.memory.semantic.knowledge["2"] = ["SEM-2"]
        self.memory.semantic.knowledge["5"] = ["SEM-5"]

        result = self.memory.to_string(
            memory_type="both",
            clip_ids=["5", "2", "9"],
        )

        self.assertIn("EP-2", result)
        self.assertIn("EP-9", result)
        self.assertIn("SEM-2", result)
        self.assertIn("SEM-5", result)

    def test_explicit_unknown_clip_is_skipped(self):
        self.memory.episodic.description["0"] = "EP-0"
        self.memory.semantic.knowledge["0"] = ["SEM-0"]

        episodic = self.memory.to_string("episodic", ["0", "99"])
        semantic = self.memory.to_string("semantic", ["0", "99"])
        both = self.memory.to_string("both", ["0", "99"])

        self.assertIn("EP-0", episodic)
        self.assertIn("SEM-0", semantic)
        self.assertIn("EP-0", both)
        self.assertIn("SEM-0", both)

    def test_empty_memory_keeps_requested_headers(self):
        self.assertTrue(self.memory.to_string("episodic").startswith("Description:"))
        self.assertTrue(self.memory.to_string("semantic").startswith("Knowledge:"))
        both = self.memory.to_string("both")
        self.assertIn("Description:", both)
        self.assertIn("Knowledge:", both)

    def test_unknown_memory_type_preserves_both_fallback(self):
        self.memory.episodic.description["0"] = "EP-0"
        self.memory.semantic.knowledge["0"] = ["SEM-0"]

        result = self.memory.to_string(memory_type="unexpected")

        self.assertIn("EP-0", result)
        self.assertIn("SEM-0", result)


if __name__ == "__main__":
    unittest.main()

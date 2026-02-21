import tempfile
import unittest
from pathlib import Path

from investigation_search import (
    EvidenceUnit,
    InvestigationEngine,
    SearchConfig,
    SourceType,
    build_manifest,
    load_build,
    write_build,
)


class InvestigationEngineTest(unittest.TestCase):
    def setUp(self):
        self.units = [
            EvidenceUnit(
                doc_id="doc-1",
                source_type=SourceType.TEXT_SENTENCE,
                content="오프라인 사전 인덱싱은 검색 지연 시간을 개선할 수 있다.",
                section_path="1.개요",
                char_start=0,
                char_end=28,
                timestamp="2026-01-01T00:00:00Z",
                confidence=0.97,
            ),
            EvidenceUnit(
                doc_id="doc-2",
                source_type=SourceType.TEXT_SENTENCE,
                content="반면 메모리가 매우 작으면 고급 인덱스 적용은 제한된다.",
                section_path="2.제한",
                char_start=0,
                char_end=29,
                timestamp="2026-01-01T00:00:00Z",
                confidence=0.93,
            ),
        ]

    def test_search_returns_evidence_and_diagnostics(self):
        engine = InvestigationEngine(self.units, build_id="demo-build")
        result = engine.search(
            "오프라인 인덱싱으로 지연 개선 가능한가",
            config=SearchConfig(top_k_per_pass=5, final_top_k=5),
            llm_config=None,
        )
        self.assertTrue(result.answer)
        self.assertTrue(result.evidence)
        self.assertIn("pass_a_support", result.diagnostics)
        self.assertEqual(result.build_id, "demo-build")

    def test_offline_build_roundtrip(self):
        manifest = build_manifest(
            self.units,
            corpus_snapshot_id="snap-1",
            parser_version="v1",
            embedding_model="dense-v1",
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            write_build(out, self.units, manifest)
            loaded_units, loaded_manifest = load_build(out)
        self.assertEqual(len(loaded_units), 2)
        self.assertEqual(loaded_manifest["knowledge_build_id"], manifest["knowledge_build_id"])


if __name__ == "__main__":
    unittest.main()

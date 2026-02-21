from __future__ import annotations

import argparse
from pathlib import Path

from .engine import InvestigationEngine
from .llm import LLMConfig
from .offline import load_build
from .schema import SearchConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Investigation Search CLI")
    parser.add_argument("--build-dir", required=True, help="Directory containing evidence_units.json and manifest.json")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--time-budget", type=int, default=120)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-model", default="llama3.1:8b")
    args = parser.parse_args()

    evidence, manifest = load_build(Path(args.build_dir))
    engine = InvestigationEngine(evidence, build_id=manifest.get("knowledge_build_id"))
    cfg = SearchConfig(top_k_per_pass=args.top_k, final_top_k=max(args.top_k, 10), time_budget_sec=args.time_budget)

    llm_cfg = None
    if args.use_llm:
        llm_cfg = LLMConfig(base_url=args.llm_url, model=args.llm_model)

    result = engine.search(args.query, config=cfg, llm_config=llm_cfg)

    print("=== ANSWER ===")
    print(result.answer)
    print("\n=== DIAGNOSTICS ===")
    for k, v in result.diagnostics.items():
        print(f"- {k}: {v}")
    print("\n=== EVIDENCE ===")
    for i, ev in enumerate(result.evidence, start=1):
        print(f"[{i}] {ev.score:.3f} {ev.verdict.value} {ev.evidence.doc_id} {ev.evidence.content}")


if __name__ == "__main__":
    main()

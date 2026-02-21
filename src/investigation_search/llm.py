from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from .schema import ScoredEvidence


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    timeout_sec: int = 45


def build_synthesis_prompt(query: str, evidence: List[ScoredEvidence]) -> str:
    lines = [
        "너는 조사형 검색 엔진의 답변 생성기다.",
        "반드시 '결론/핵심근거/반례/한계' 4개 섹션을 한국어로 작성하라.",
        f"질문: {query}",
        "근거 후보:",
    ]
    for idx, item in enumerate(evidence, start=1):
        lines.append(
            (
                f"[{idx}] score={item.score:.3f} verdict={item.verdict.value} "
                f"doc={item.evidence.doc_id} sec={item.evidence.section_path} "
                f"text={item.evidence.content}"
            )
        )
    return "\n".join(lines)


def _call_ollama_generate(prompt: str, cfg: LLMConfig) -> str:
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        url=f"{cfg.base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as res:
        body = json.loads(res.read().decode("utf-8"))
    return body.get("response", "").strip()


def synthesize_answer(
    query: str,
    evidence: List[ScoredEvidence],
    llm_config: Optional[LLMConfig],
) -> Dict[str, str]:
    if llm_config is None:
        return {
            "answer": evidence[0].evidence.content if evidence else "근거를 찾지 못했습니다.",
            "mode": "heuristic",
        }

    prompt = build_synthesis_prompt(query, evidence)
    try:
        text = _call_ollama_generate(prompt, llm_config)
        if text:
            return {"answer": text, "mode": "local_llm"}
        return {
            "answer": evidence[0].evidence.content if evidence else "근거를 찾지 못했습니다.",
            "mode": "heuristic_empty_llm",
        }
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {
            "answer": evidence[0].evidence.content if evidence else "근거를 찾지 못했습니다.",
            "mode": "heuristic_fallback",
        }

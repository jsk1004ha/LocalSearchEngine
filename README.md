# Local Investigation Search Engine

개인용 PC 환경에서 동작하는 **조사형 검색 엔진(Investigation Search Engine)** 구현체입니다.

이번 버전은 기존 스캐폴드에서 확장되어, 실제 고성능 운영을 위한 다음 기능을 포함합니다.

- 하이브리드 점수화(lexical + dense-like + rerank)
- 멀티패스 탐색(정방향/반증/경계조건)
- 시간 예산 기반 검색 중단/진단
- 오프라인 빌드 ID + 체크섬 + 재로드
- **로컬 LLM(Ollama API) 연동**으로 근거 기반 답변 합성

## 핵심 개념

검색 결과는 항상 다음을 목표로 합니다.

1. 결론(Answer)
2. 근거(Evidence spans)
3. 조건 충족 이유(why it matches)
4. 반례/모순(Disconfirming evidence)

## 설치/실행 (로컬)

### 1) 오프라인 빌드 산출물 준비

빌드 디렉터리에는 아래 두 파일이 있어야 합니다.

- `evidence_units.json`
- `manifest.json`

오프라인 빌드는 `build_manifest()` + `write_build()`로 생성합니다.

### 2) CLI 검색

```bash
PYTHONPATH=src python -m investigation_search.cli \
  --build-dir ./my_build \
  --query "오프라인 인덱싱으로 지연 개선 가능한가" \
  --top-k 10 \
  --time-budget 120
```

### 3) 로컬 LLM 연결 (Ollama)

Ollama가 로컬에서 실행 중일 때:

```bash
PYTHONPATH=src python -m investigation_search.cli \
  --build-dir ./my_build \
  --query "메모리 제약 환경에서 성능 유지 전략은?" \
  --use-llm \
  --llm-url http://127.0.0.1:11434 \
  --llm-model llama3.1:8b
```

LLM 연결 실패 시 자동으로 heuristic 답변으로 폴백합니다.

## Python API 예시

```python
from investigation_search import (
    EvidenceUnit,
    InvestigationEngine,
    LLMConfig,
    SearchConfig,
    SourceType,
)

units = [
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

engine = InvestigationEngine(units, build_id="demo-build")
result = engine.search(
    "오프라인 인덱싱으로 지연 개선 가능한가",
    config=SearchConfig(top_k_per_pass=8, final_top_k=12, time_budget_sec=120),
    llm_config=LLMConfig(base_url="http://127.0.0.1:11434", model="llama3.1:8b"),
)

print(result.answer)
print(result.diagnostics)
```

## 프로젝트 구조

- `src/investigation_search/schema.py`: 데이터 모델 + 검색 설정
- `src/investigation_search/retrieval.py`: 하이브리드 점수화 + 패스별 회수
- `src/investigation_search/engine.py`: 전체 탐색/시간예산/답변 합성 orchestration
- `src/investigation_search/llm.py`: 로컬 LLM(Ollama API) 연결 및 폴백
- `src/investigation_search/offline.py`: 오프라인 빌드/매니페스트/재로드
- `src/investigation_search/cli.py`: 실행용 CLI
- `agent.md`: 저장소 작업 원칙
- `VERSION`: 프로젝트 버전

## 한계 및 다음 단계

현재 dense는 외부 임베딩 모델 없이 동작하는 dense-like cosine 방식입니다. 실운영 고성능화 시:

- 실제 임베딩 모델(e5/bge 등) + ANN 인덱스(FAISS/HNSW)
- 크로스 인코더 reranker
- contradiction 분류기 별도 학습/서빙
- 증분 인덱싱 파이프라인

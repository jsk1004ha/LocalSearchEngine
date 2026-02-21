# Local Investigation Search Engine

개인용 PC 환경에서 동작하는 **조사형 검색 엔진(Investigation Search Engine)**의 최소 구현체입니다.
목표는 단순 유사도 검색이 아니라 아래 4가지를 함께 반환하는 것입니다.

1. 결론(Answer)
2. 근거(Evidence spans)
3. 출처(Citations)
4. 반례/모순(Disconfirming evidence)

## 구현된 핵심 설계

- **오프라인 지식 빌드**
  - `knowledge_build_id` 생성(코퍼스 스냅샷/파서 버전/임베딩 버전 결합)
  - evidence 산출물 체크섬 포함 매니페스트 생성
  - `previous_build_dir` 기반 증분 임베딩 재사용(reused/new 벡터 통계 기록)
  - 해시 기반 샤딩 산출물(`shards/shard_xxx`) 생성 및 로딩 지원
- **근거 단위 스키마**
  - `text_sentence`, `table_cell`, `caption`, `ocr_text` source type 지원
  - 문서 위치(`section_path`, `char_start`, `char_end`)와 신뢰도(`confidence`) 저장
  - 모든 `ScoredEvidence`/`SearchResult`에 `SourceCitation` 포함(`answer_sources`, `sources`)
- **온라인 멀티패스 하이브리드 검색**
  - Pass A: 정방향
  - Pass B: 반증(예외/제한/반대 근거 탐색)
  - Pass C: 경계조건(대상/시점/조건 변경)
  - BM25 + Dense + lexical + RRF 융합 스코어
  - cross-encoder 스타일 reranker 후 최종 점수 반영
- **모순 검출 전용 단계**
  - `ContradictionDetector` 인터페이스와 기본 로컬 heuristic detector
  - NLI/ONNX 런타임 어댑터로 전용 모델 연동 가능
- **진단 정보(Diagnostics)**
  - 각 pass hit 수, 실행 시간, time budget, 반례 근거 부족 여부 표기
  - rerank 적용/생략 사유, contradiction detector 오버라이드 수, cache hit/miss 포함
- **검색 중 온라인 학습**
  - 검색 결과의 stage score를 이용해 BM25/dense/lexical/RRF 가중치를 점진 업데이트
  - query token boost를 누적 학습하여 다음 검색의 lexical/BM25 스코어에 반영
  - 학습 상태 파일(JSON) 로드/저장 지원
- **Ollama 로컬 모델 연동**
  - `ollama:<model>` 임베딩 모델명으로 dense 임베딩을 Ollama `/api/embed`로 생성
  - `OllamaRerankerAdapter`, `OllamaContradictionDetector`로 reranker/NLI 연결
- **엔터티 정규화/집계**
  - 이름/닉네임/아이디/메일 마스킹 표기를 정규화하고 alias 후보를 query expansion에 반영
  - alias 확장 점수 페널티로 오탐을 제어하고 evidence metadata의 `canonical_entity_id` 기준 그룹 집계 지원
- **문서 파서 확장**
  - TXT/MD/CSV + PDF(text/table) + 이미지 OCR 파싱
  - `DocumentParser`/`parse_documents()`로 근거 단위 자동 생성

## 의존성

기본 검색 경로는 표준 라이브러리만으로 동작하며, 아래는 선택 의존성입니다.

- `numpy`
- `sentence-transformers`
- `hnswlib` 또는 `faiss-cpu` (없으면 exact fallback)
- PDF 파서: `pdfplumber` 또는 `pypdf`
- OCR: `pillow`, `pytesseract` (+ 시스템 tesseract 설치)
- Ollama 연동: 로컬 `ollama serve` 실행

## 빠른 시작

```bash
python - <<'PY'
from investigation_search import EvidenceUnit, InvestigationEngine, SourceType

units = [
    EvidenceUnit(
        doc_id="doc-1",
        source_type=SourceType.TEXT_SENTENCE,
        content="로컬 검색 파이프라인은 오프라인 사전 인덱싱으로 지연을 줄일 수 있다.",
        section_path="1.개요",
        char_start=0,
        char_end=36,
        timestamp="2026-01-01T00:00:00Z",
        confidence=0.95,
    ),
    EvidenceUnit(
        doc_id="doc-2",
        source_type=SourceType.TEXT_SENTENCE,
        content="하지만 작은 메모리 환경에서는 고차원 인덱스가 제한될 수 있다.",
        section_path="2.제한",
        char_start=0,
        char_end=34,
        timestamp="2026-01-01T00:00:00Z",
        confidence=0.92,
    ),
]

engine = InvestigationEngine(units, build_id="example-build")
result = engine.search("오프라인 인덱싱으로 지연 개선 가능한가")

print("answer:", result.answer)
print("answer_sources:", [s.citation_id for s in result.answer_sources])
print("sources:", [s.citation_id for s in result.sources])
print("diagnostics:", result.diagnostics)
print("contradictions:", len(result.contradictions))
PY
```

> 위 예시는 로컬 패키지 경로를 위해 `PYTHONPATH=src` 또는 설치 후 실행해야 합니다.


## 임베딩/ANN 권장 설정 (CPU)

- 권장 임베딩 모델
  - `intfloat/multilingual-e5-small` (384d, 한국어/영어 혼합 질의에 안정적)
  - `BAAI/bge-small-en-v1.5` (384d, 영문 중심 코퍼스)
- 권장 인덱스
  - 기본: HNSW (`M=32`, `ef_construction=200`, `ef_search=64`)
  - 대규모 데이터(수백만 단위): FAISS IVF (`nlist=256` 이상)
- 메모리 가이드(대략)
  - 임베딩 행렬: `문서수 N * 384 * 4 bytes` (float32)
    - 예: 100,000 span ≈ 146MB
  - HNSW 그래프 오버헤드: 데이터 분포에 따라 임베딩 메모리의 약 1.2~2.0배 추가
  - 운영시 여유 메모리(파이썬 런타임/버퍼 포함)로 최소 2~3배를 권장

오프라인 빌드 시 `write_build()`는 다음 산출물을 생성합니다.

- `evidence_units.json`
- `evidence_embeddings.npy`
- `ann_index.bin`(또는 fallback 시 `ann_index.npy`) + `ann_index.meta.json`
- `manifest.json`
- (옵션) `shards/shard_xxx/*` 분할 아티팩트

온라인 검색은 lexical 후보와 ANN 후보를 합쳐 `doc_id + (char_start, char_end)` 기준으로 중복 병합한 뒤 재랭킹합니다.

## Ollama 연동 예시

```python
from investigation_search import (
    InvestigationEngine,
    OllamaContradictionDetector,
    OllamaRerankerAdapter,
)

engine = InvestigationEngine(
    units,
    embedding_model="ollama:nomic-embed-text",
    reranker=OllamaRerankerAdapter(model="llama3.1:8b"),
    contradiction_detector=OllamaContradictionDetector(model="llama3.1:8b"),
    online_learning=True,
    learning_state_path="artifacts/learning_state.json",
)
```

## 프로젝트 구조

- `src/investigation_search/schema.py`: 근거/판정/result 모델
- `src/investigation_search/offline.py`: 오프라인 빌드 및 매니페스트
- `src/investigation_search/retrieval.py`: BM25 + dense + lexical 하이브리드 후보 결합/중복 병합/판정
- `src/investigation_search/bm25.py`: BM25 인덱스/검색
- `src/investigation_search/learning.py`: 검색 중 온라인 학습(가중치/토큰 boost)
- `src/investigation_search/ollama.py`: Ollama API 클라이언트 + reranker/contradiction 어댑터
- `src/investigation_search/embedding.py`: 로컬 임베딩 모델 로더/인코딩
- `src/investigation_search/index_ann.py`: ANN 인덱스 빌드/검색/저장/로딩
- `src/investigation_search/contradiction.py`: 모순 검출 전용 detector 인터페이스/어댑터
- `src/investigation_search/parser.py`: PDF/표/OCR 문서 파서
- `src/investigation_search/engine.py`: 시간 예산 + cache + shard-aware 조사형 검색 실행기
- `VERSION`: 프로젝트 버전
- `agent.md`: 이 저장소에서 작업하는 에이전트용 운영 규칙

## 고도화 상태

아래 항목은 현재 구현에 반영되어 있습니다.

- BM25 + Dense + reranker 하이브리드
- 문서 파서 확장(PDF/표/OCR)
- contradiction detection 전용 모델 인터페이스 + 기본 detector
- 쿼리 캐시 + 샤딩 + 증분 빌드

# Local Investigation Search Engine

개인용 PC 환경에서 동작하는 **조사형 검색 엔진(Investigation Search Engine)**의 최소 구현체입니다.
목표는 단순 유사도 검색이 아니라 아래 4가지를 함께 반환하는 것입니다.

1. 결론(Answer)
2. 근거(Evidence spans)
3. 조건 충족 이유(Why it matches)
4. 반례/모순(Disconfirming evidence)

## 구현된 핵심 설계

- **오프라인 지식 빌드**
  - `knowledge_build_id` 생성(코퍼스 스냅샷/파서 버전/임베딩 버전 결합)
  - evidence 산출물 체크섬 포함 매니페스트 생성
- **근거 단위 스키마**
  - `text_sentence`, `table_cell`, `caption`, `ocr_text` source type 지원
  - 문서 위치(`section_path`, `char_start`, `char_end`)와 신뢰도(`confidence`) 저장
- **온라인 멀티패스 검색**
  - Pass A: 정방향
  - Pass B: 반증(예외/제한/반대 근거 탐색)
  - Pass C: 경계조건(대상/시점/조건 변경)
- **진단 정보(Diagnostics)**
  - 각 pass hit 수, 실행 시간, time budget, 반례 근거 부족 여부 표기

## 의존성

dense 검색 경로를 사용하려면 아래 패키지가 필요합니다.

- `numpy`
- `sentence-transformers`
- `hnswlib` 또는 `faiss-cpu` (없으면 exact fallback)

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

온라인 검색은 lexical 후보와 ANN 후보를 합쳐 `doc_id + (char_start, char_end)` 기준으로 중복 병합한 뒤 재랭킹합니다.

## 프로젝트 구조

- `src/investigation_search/schema.py`: 근거/판정/result 모델
- `src/investigation_search/offline.py`: 오프라인 빌드 및 매니페스트
- `src/investigation_search/retrieval.py`: lexical + dense 후보 결합/중복 병합/판정
- `src/investigation_search/embedding.py`: 로컬 임베딩 모델 로더/인코딩
- `src/investigation_search/index_ann.py`: ANN 인덱스 빌드/검색/저장/로딩
- `src/investigation_search/engine.py`: 시간 예산 기반 조사형 검색 실행기
- `VERSION`: 프로젝트 버전
- `agent.md`: 이 저장소에서 작업하는 에이전트용 운영 규칙

## 한계

현재 구현은 lexical 기반 최소 프로토타입입니다. 실제 고성능 운영에서는 다음 확장이 필요합니다.

- BM25 + Dense + reranker 하이브리드
- 문서 파서 확장(PDF/표/OCR)
- contradiction detection 전용 모델
- 캐시/샤딩/증분 빌드

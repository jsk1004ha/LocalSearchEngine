# Local Investigation Search Engine

로컬 PC 환경에서 돌아가는 **조사형(Investigation) 검색 엔진**입니다.  
단순 “유사 문장 찾기”가 아니라, 질문에 대해 **결론 + 근거 + 출처 + 반례/모순 + 설명가능한 진단**을 함께 반환하는 것을 목표로 합니다.

## TL;DR

- 오프라인 인덱싱(임베딩/ANN/BM25) + 온라인 멀티패스 하이브리드 검색
- 모든 결과에 `SourceCitation`(출처) 포함
- 모순(contradiction) 전용 단계 + reranker 단계
- 검색 DSL(필터) + explain/하이라이트 + 평가 하니스
- 온라인 학습(가중치/토큰 boost) + 사용자 검색정보 삭제 API
- 로컬 근거 부족 시 DuckDuckGo `web_snippet` fallback(옵션)

## 목차

- [왜 필요한가](#왜-필요한가)
- [주요 기능](#주요-기능)
- [빠른 시작](#빠른-시작)
- [설치](#설치)
- [사용법](#사용법)
- [검색 DSL](#검색-dsl)
- [검색 모드](#검색-모드)
- [지식도서관](#지식도서관)
- [설정](#설정)
- [개인정보/데이터](#개인정보데이터)
- [문제 해결](#문제-해결)
- [평가](#평가)
- [지원](#지원)
- [라이선스](#라이선스)
- [변경 로그](#변경-로그)

## 왜 필요한가

조사/리서치 상황에서는 “가장 그럴듯한 문장”보다 아래가 더 중요합니다.

- **무엇을 근거로 결론을 냈는지**(출처/스팬/맥락)
- **반례나 조건부 예외**가 있는지(모순/반증 탐색)
- **왜 그 근거가 선택됐는지**(설명가능한 점수/진단)
- 운영 관점에서 **재현 가능**하고 **삭제 가능**해야 함(오프라인 빌드, 개인정보 제어)

## 주요 기능

- **오프라인 빌드**
  - evidence units(`evidence_units.json`)
  - embeddings(`evidence_embeddings.npy`) + ANN(`ann_index.*`)
  - BM25(`bm25_index.json`) 저장/로드
  - 샤딩 아티팩트(`shards/shard_xxx/*`)
  - 증분 임베딩 재사용(`previous_build_dir`)
- **온라인 멀티패스 하이브리드 검색**
  - Pass A(정방향) / Pass B(반증) / Pass C(경계조건)
  - BM25 + dense + lexical + RRF 융합 점수
  - 남은 time budget에 따라 후보 수 동적 조절
  - reranker는 상위 후보만 선택 적용(`max_rerank_candidates`)
- **모순 검출 단계**
  - `ContradictionDetector` 인터페이스
  - 기본 heuristic detector
  - ONNX 또는 Ollama 기반 detector 어댑터
- **출처(Citations)**
  - 모든 `ScoredEvidence`/`SearchResult`에 `SourceCitation` 포함
  - `answer_sources`, `sources` 제공
- **Explain/하이라이트**
  - `engine.explain(result)`로 근거별 score/why/stage를 구조화 출력
  - `render_result_text()`로 하이라이트된 텍스트 뷰 출력
- **검색 DSL**
  - `doc:`, `source:`, `-source:`, `after:`, `before:`, `section:`
- **검색 모드**
  - `investigation`(조사), `reporter`(기자), `fbi`(OSINT), `collection`(자료), `sniper`(핵심), `rumor`(찌라시), `library`(신뢰)
- **지식도서관**
  - 모든 모드 결과를 로컬 저장소에 자동 저장(옵션)
- **온라인 학습 + 삭제**
  - stage score 기반 가중치/토큰 boost 점진 업데이트
  - `delete_user_search_data()`로 캐시/학습 상태 즉시 삭제
- **웹 fallback (옵션)**
  - 로컬 근거가 부족할 때 DuckDuckGo 결과를 `web_snippet` 근거로 병합
  - DSL에서 `-source:web_snippet`로 외부 결과 제외 가능
  - 기본적으로 웹 검색은 **별도 파이썬 프로세스(서브프로세스)**에서 수행(최소 격리). 강한 격리가 필요하면 VM/Docker/Windows Sandbox 권장

## 빠른 시작

로컬 패키지 경로를 위해 `PYTHONPATH=src`로 실행합니다.

```powershell
$env:PYTHONPATH="src"
python - <<'PY'
from investigation_search import EvidenceUnit, InvestigationEngine, SourceType

units = [
    EvidenceUnit(
        doc_id="doc-1",
        source_type=SourceType.TEXT_SENTENCE,
        content="오프라인 인덱싱은 검색 지연을 줄일 수 있다.",
        section_path="1.개요",
        char_start=0,
        char_end=23,
        timestamp="2026-01-01T00:00:00Z",
        confidence=0.95,
    ),
    EvidenceUnit(
        doc_id="doc-2",
        source_type=SourceType.TEXT_SENTENCE,
        content="단, 메모리 환경에서는 인덱스 크기 제한이 발생할 수 있다.",
        section_path="2.제한",
        char_start=0,
        char_end=35,
        timestamp="2026-01-01T00:00:00Z",
        confidence=0.92,
    ),
]

engine = InvestigationEngine(units, build_id="example-build")
result = engine.search('오프라인 인덱싱으로 지연 개선 가능한가')

print("answer:", result.answer)
print("answer_sources:", [s.citation_id for s in result.answer_sources])
print("sources:", [s.citation_id for s in result.sources])
print("contradictions:", len(result.contradictions))
print("diagnostics_keys:", sorted(result.diagnostics.keys()))
PY
```

## 설치

### 필수

- Python 3.11+

### 권장 (requirements)

이 저장소는 기능 단위로 `requirements-*.txt`를 제공합니다.

```powershell
# 전체 기능(검색 + 문서파서 + TUI + publish)
pip install -r requirements.txt

# 또는 필요한 것만:
pip install -r requirements-search.txt     # 하이브리드 검색(embeddings/ANN) + core
pip install -r requirements-docparse.txt   # PDF/표/OCR 파서
pip install -r requirements-tui.txt        # TUI(Textual)
pip install -r requirements-publish.txt    # PDF publish
```

### 선택(기능별)

- Dense 임베딩/ANN: `numpy`, `sentence-transformers`, `hnswlib` 또는 `faiss-cpu`
- PDF: `pdfplumber` 또는 `pypdf`
- OCR: `pillow`, `pytesseract` (+ 시스템 tesseract 설치)
- Ollama: 로컬 `ollama serve`
- 형태소 분석(옵션): `konlpy` (Analyzer가 자동으로 있으면 사용)

예시:

```powershell
pip install -U numpy sentence-transformers hnswlib
pip install -U pdfplumber pillow pytesseract
```

## 사용법

### 0) CLI/TUI

패키지 설치 없이 실행할 때는 `PYTHONPATH=src`가 필요합니다.

```powershell
$env:PYTHONPATH="src"
$env:INVESTIGATION_SEARCH_AUTO_INSTALL="1"  # (옵션) 필요한 파이썬 의존성 자동 설치

# TUI 실행 (오프라인 빌드 사용 권장)
python -m investigation_search tui --build-dir artifacts/build --enable-knowledge-library

# 지식도서관 보기용 HTML export
python -m investigation_search library export --out-dir artifacts/library_export --format html

# 지식도서관 publish(zip)
python -m investigation_search library publish --out artifacts/library_bundle.zip --format zip
```

### 1) 오프라인 빌드

```python
from pathlib import Path
from investigation_search import build_manifest, write_build

manifest = build_manifest(units, corpus_snapshot_id="snap-2026-01-01", parser_version="2.0", embedding_model="intfloat/multilingual-e5-small")
write_build(Path("artifacts/build"), units, manifest, shard_count=4, previous_build_dir=None)
```

### 2) 검색

```python
result = engine.search("조건 변경 시 성능 향상 가능한가", top_k_per_pass=5, time_budget_sec=120)
```

### 3) Explain/렌더링

```python
from investigation_search import render_result_text

print(render_result_text(result, query="조건 변경 시 성능 향상 가능한가"))
print(engine.explain(result, max_items=5))
```

## 검색 DSL

예시:

- `doc:report_2026 source:table after:2026-01-01 "매출 증가"`
- `-source:ocr "내부 문서에서만 찾기"`
- `section:page/3 "핵심 결론"`

지원 연산자:

- `doc:<doc_id>`
- `source:<text|table|caption|ocr|web_snippet|...>`
- `-source:<...>`
- `after:<YYYY-MM-DD|ISO8601>`
- `before:<YYYY-MM-DD|ISO8601>`
- `section:<prefix>`

## 검색 모드

`engine.search()`에 `mode=`를 지정합니다.

```python
result = engine.search("주제", mode="investigation")  # 기본 조사 모드
result = engine.search("주제", mode="reporter")        # 폭넓은 수집/요약
result = engine.search("주제", mode="fbi")             # OSINT 중심(공개출처 기반)
result = engine.search("주제", mode="collection")      # 자료/링크 중심
result = engine.search("주제", mode="sniper")          # 1개 핵심 근거만
result = engine.search("주제", mode="rumor")           # 다양한 주장 종합(진위 미확인 라벨)
result = engine.search("주제", mode="library")         # 신뢰 소스 우선(기본: 로컬)
```

주의:

- `fbi` 모드는 **공개 출처(OSINT) 기반**으로만 수집합니다. 스캔/침투/우회 같은 행위는 지원하지 않습니다.
- `fbi` 모드는 `osint.graph`(관계도)와 `osint.timeline`(날짜 이벤트)을 diagnostics 및 지식도서관에 저장합니다.
- `rumor` 모드는 진위와 무관하게 다양한 관점을 모으되, 결과를 **미확인**으로 표시합니다.

## 지식도서관

모든 검색 결과/근거를 하나의 로컬 저장소에 쌓아두고, 나중에 찾기 쉽게 보관합니다.

```python
from investigation_search import InvestigationEngine

engine = InvestigationEngine(
    units,
    enable_knowledge_library=True,
    knowledge_library_dir="artifacts/knowledge_library",
)
result = engine.search("주제", mode="reporter")
```

저장되는 것:

- `artifacts/knowledge_library/sessions/<session_id>.json`: 검색 결과(모드/쿼리/출처/근거/진단)
- `artifacts/knowledge_library/sessions.jsonl`: 세션 인덱스
- `artifacts/knowledge_library/evidence/web_snippets.jsonl`: 웹 스니펫 근거(있을 때)
- (옵션) `artifacts/knowledge_library/osint/<session_id>.json`: OSINT 아티팩트(모드에 따라)

### 보기(Export)

지식도서관은 JSON이지만, 보기 쉽게 **HTML/Markdown으로 export**할 수 있습니다.

```powershell
$env:PYTHONPATH="src"

# HTML export (브라우저로 index.html 열기)
python -m investigation_search library export --out-dir artifacts/library_export --format html

# (옵션) 로컬 HTTP 서버로 보기
python -m investigation_search library serve --dir artifacts/library_export --port 8765

# Markdown export
python -m investigation_search library export --out-dir artifacts/library_md --format md
```

### Publish

공유/보관 용도로 zip(정적 HTML/MD + raw JSON)을 생성할 수 있습니다.

```powershell
$env:PYTHONPATH="src"
python -m investigation_search library publish --out artifacts/library_bundle.zip --format zip
```

PDF publish는 선택 기능이며 `fpdf2`와 유니코드 폰트가 필요합니다.

```powershell
pip install -r requirements-publish.txt
python -m investigation_search library publish --out artifacts/library.pdf --format pdf --max-sessions 30
```

## 설정

엔진 생성 시 주로 쓰는 옵션:

- `retrieval_options`: `confidence/source_type/timestamp` 품질 prior, OCR 임계값, recency boost 등 (`src/investigation_search/retrieval.py`)
- `max_rerank_candidates`: rerank 적용 후보 수 상한 (`src/investigation_search/engine.py`)
- `online_learning`, `learning_state_path`: 검색 중 학습 on/off 및 상태 파일 (`src/investigation_search/learning.py`)
- `enable_web_fallback`, `web_search_provider`: DuckDuckGo fallback on/off (`src/investigation_search/websearch.py`)
- `enable_knowledge_library`, `knowledge_library_dir`: 지식도서관 저장 on/off (`src/investigation_search/library.py`)
- `trusted_domains`: `library` 모드에서 허용할 웹 도메인 allowlist (`src/investigation_search/engine.py`)

## 개인정보/데이터

- 사용자 검색 정보는 **메모리 캐시**와 **학습 상태 JSON**에만 저장됩니다(설정 시).
- 언제든 아래로 즉시 삭제할 수 있습니다:

```python
engine.delete_user_search_data(delete_learning_state_file=True)
```

- 지식도서관까지 함께 삭제하려면:

```python
engine.delete_user_search_data(delete_learning_state_file=True, delete_knowledge_library=True)
```

- CLI에서 지식도서관 전체 삭제:

```powershell
$env:PYTHONPATH="src"
python -m investigation_search library delete --yes
```

- 웹 fallback은 외부로 쿼리를 전송합니다. 개인정보가 민감하면:
  - `enable_web_fallback=False` 또는
  - DSL로 `-source:web_snippet` 사용
  - (옵션) 웹 fallback 격리 해제: `enable_web_sandbox=False` 또는 TUI에서 `--no-web-sandbox`

## 문제 해결

자주 발생하는 이슈는 `TROUBLESHOOTING.md`에 정리했습니다.

- `sentence-transformers가 필요합니다` 등 옵션 의존성 관련
- HuggingFace 모델 다운로드 실패(프록시/오프라인 환경)
- OCR/PDF/Ollama 연동 실패
- DuckDuckGo 접근 차단/타임아웃

## 평가

`evaluate_engine()`로 회귀 테스트용 지표를 계산합니다.

```python
from investigation_search import EvaluationCase, evaluate_engine

cases = [
    EvaluationCase(
        query="처리 속도 개선",
        relevant_doc_ids=("doc-1",),
        expect_contradiction=True,
        top_k=5,
    )
]
report = evaluate_engine(engine, cases)
print(report.metrics)
```

## 지원

- 동작/설계 질문: 이 저장소의 이슈 트래커에 재현 가능한 예제와 함께 남겨주세요.
- 보안/개인정보 관련: 민감 데이터는 첨부하지 말고, 재현 가능한 최소 사례만 공유하세요.

## 라이선스

Apache-2.0 (see `LICENSE`)

## 변경 로그

`CHANGELOG.md` 참고 (`VERSION`가 단일 소스)

# Web Investigation Search Engine

로컬에서 실행되는 **웹 전용 조사형(Investigation) 검색 엔진**입니다.  
이 프로젝트는 로컬 파일/문서 코퍼스 검색을 제공하지 않으며, 온라인 웹 검색 기반으로만 동작합니다.

## TL;DR

- DuckDuckGo 기반 **웹 멀티패스 하이브리드 검색**
- (옵션) 결과 URL 페이지 본문 fetch(HTML/PDF)
- 모든 결과에 `SourceCitation` 포함
- 모순(contradiction) 단계 + reranker + explain/diagnostics
- 지식도서관(Knowledge Library) 저장/내보내기/게시 지원

## 빠른 시작

```powershell
$env:PYTHONPATH="src"
$env:INVESTIGATION_SEARCH_AUTO_INSTALL="1"  # (옵션)

# 웹 검색 TUI 실행
python -m investigation_search tui --enable-knowledge-library --web-fetch
```

## 설치

- Python 3.11+

```powershell
pip install -r requirements-core.txt
pip install -r requirements-tui.txt
```

필요 시:

```powershell
pip install -r requirements-publish.txt
```

## 사용법

### TUI

```powershell
$env:PYTHONPATH="src"
python -m investigation_search tui --web-fetch --enable-knowledge-library
```

주요 옵션:

- `--mode`: 검색 모드
- `--top-k`: 패스별 후보 수
- `--time-budget`: 검색 시간 예산(초)
- `--no-web-sandbox`: 웹 검색 서브프로세스 격리 비활성화
- `--web-fetch`: 검색 결과 페이지 본문 fetch 활성화
- `--web-fetch-pages`: fetch 페이지 수 제한

### Knowledge Library

```powershell
python -m investigation_search library list
python -m investigation_search library export --out-dir artifacts/library_export --format html
python -m investigation_search library publish --out artifacts/library_bundle.zip --format zip
```

## 설계 원칙

- 웹 검색 결과에 대해 `결론 + 근거 + 반례 + 진단`을 함께 제공
- 출처 추적 가능한 evidence/citation 구조 유지
- 시간 예산 내에서 안정적으로 동작

## 라이선스

MIT License. 자세한 내용은 `LICENSE`를 참고하세요.

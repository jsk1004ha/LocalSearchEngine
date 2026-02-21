# Troubleshooting

## `ModuleNotFoundError: No module named 'investigation_search'`

- `PYTHONPATH=src`가 필요합니다.
- PowerShell: `$env:PYTHONPATH="src"`

## `sentence-transformers가 필요합니다...`

- Dense 임베딩 경로를 사용 중입니다.
- 설치: `pip install -U sentence-transformers numpy`

## HuggingFace 모델 다운로드 실패 / 프록시 / 오프라인 환경

- `sentence-transformers` 모델은 기본적으로 HuggingFace Hub에서 다운로드합니다.
- 네트워크가 차단된 환경에서는:
  - 사전 다운로드 후 캐시 디렉터리를 설정하거나
  - `embedding_model="ollama:<model>"`로 Ollama 임베딩 사용을 고려하세요.

## OCR 관련 오류

- `pytesseract`는 파이썬 패키지 외에 시스템 `tesseract` 설치가 필요합니다.
- `pillow`도 필요합니다.

## PDF 파서 관련 오류

- `pdfplumber` 또는 `pypdf`를 설치하세요.
- 표 추출은 `pdfplumber`가 더 유리합니다.

## Ollama 연결 실패

- `ollama serve` 실행 상태 확인
- `OLLAMA_BASE_URL`이 다른 경우 환경변수로 지정
  - 예: `$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"`

## DuckDuckGo web fallback 타임아웃/접근 차단

- 네트워크 정책/방화벽/프록시 환경에서 차단될 수 있습니다.
- 비활성화:
  - `InvestigationEngine(..., enable_web_fallback=False)`
  - 또는 검색 DSL로 `-source:web_snippet`


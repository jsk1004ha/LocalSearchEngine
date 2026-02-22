<p align="center">
  <h1 align="center">Web Investigation Search Engine</h1>
  <p align="center">로컬에서 실행되는 <b>웹 조사형(Investigation) 검색 엔진</b> (온라인 웹 검색 기반)</p>
  <p align="center">
    <a href="#quick-start">빠른 시작</a> ·
    <a href="#install">설치</a> ·
    <a href="#usage">사용법</a> ·
    <a href="#config">설정</a> ·
    <a href="#troubleshooting">트러블슈팅</a>
  </p>
  <p align="center">
    <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
    <img alt="version" src="https://img.shields.io/badge/version-0.9.0-blue">
    <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
    <img alt="status" src="https://img.shields.io/badge/status-experimental-orange">
  </p>
</p>

## TL;DR

- **SearxNG 기반 웹 멀티패스 하이브리드 검색** (로컬 메타서치)
- (옵션) 결과 URL **본문 fetch + 파싱**(HTML/PDF/이미지 OCR) + **사이트 내부 링크 크롤**
- 모든 결과에 `SourceCitation` 포함 (추적 가능한 근거 구조)
- 모순(contradiction) 단계 + reranker + explain/diagnostics
- 지식도서관(Knowledge Library) 저장/내보내기/게시

<details>
  <summary><b>Table of Contents</b></summary>

- [빠른 시작](#quick-start)
- [설치](#install)
  - [사전 준비](#prerequisites)
  - [의존성 설치](#dependencies)
- [사용법](#usage)
  - [TUI](#tui)
  - [검색 모드](#modes)
  - [Knowledge Library](#knowledge-library)
- [설정](#config)
  - [환경 변수](#env-vars)
  - [웹 검색/격리](#web-sandbox)
- [개발](#development)
- [기여](#contributing)
- [트러블슈팅](#troubleshooting)
- [버전/변경 이력](#changelog)
- [라이선스](#license)

</details>

<a id="quick-start"></a>

## 빠른 시작

> 이 프로젝트는 **온라인 웹 검색 기반**으로 동작합니다. 로컬 SearxNG가 필요합니다(기본: `http://127.0.0.1:8080/search`).

```powershell
$env:PYTHONPATH="src"
$env:INVESTIGATION_SEARCH_AUTO_INSTALL="1"  # (옵션) 실행 중 누락된 선택 의존성 자동 설치

# 웹 검색 TUI 실행 (Textual)
python -m investigation_search tui --enable-knowledge-library --web-fetch
```

<a id="install"></a>

## 설치

<a id="prerequisites"></a>

### 사전 준비

- Python 3.11+
- (권장) 로컬 SearxNG 실행: 기본 엔드포인트는 `http://127.0.0.1:8080/search`
- (옵션) OCR을 쓰려면 시스템 `tesseract` 설치 필요 (`pytesseract`는 래퍼)

<a id="dependencies"></a>

### 의존성 설치

최소 런타임:

```powershell
pip install -r requirements-core.txt
```

TUI:

```powershell
pip install -r requirements-tui.txt
```

선택 기능(필요할 때만):

- 검색(임베딩/ANN): `pip install -r requirements-search.txt`
- 문서 파싱(PDF/OCR): `pip install -r requirements-docparse.txt`
- 게시(PDF/zip): `pip install -r requirements-publish.txt`
- 한국어 형태소(주의: Java/JPype 필요): `pip install -r requirements-ko-morph.txt`

<a id="usage"></a>

## 사용법

<a id="tui"></a>

### TUI

```powershell
$env:PYTHONPATH="src"
python -m investigation_search tui --web-fetch --enable-knowledge-library
```

도움말:

```powershell
python -m investigation_search tui --help
```

주요 옵션:

- `--mode`: 기본 검색 모드
- `--top-k`: 패스별 후보 수
- `--time-budget`: 시간 예산(초)
- `--no-web-sandbox`: 웹 검색 서브프로세스 격리 비활성화
- `--web-fetch`: 결과 URL 본문 fetch 활성화
- `--web-fetch-pages`: fetch 페이지 수 제한
- `--web-fetch-crawl-depth`: 사이트 내부 링크 추가 크롤 깊이(0~2)
- `--web-fetch-workers`: 웹 페이지 fetch 병렬 워커 수

<a id="modes"></a>

### 검색 모드

| mode | 의도 |
| --- | --- |
| `investigation` | 결론 + 근거 + 반례 + 진단을 균형 있게 제공 |
| `reporter` | 폭넓고 상세한 자료 수집(웹 fetch 강화) |
| `fbi` | OSINT 성격의 심층 수집(메타/아카이브/보안 문서 탐색 포함) |
| `collection` | 자료(논문/데이터셋/레포 등) 중심 수집 |
| `sniper` | 단 하나의 가장 강한 근거만 최대한 빠르게 선택 |
| `rumor` | 진위와 무관하게 다양한 주장/의견 수집(라벨링 포함) |
| `library` | trusted-first 집계(정책에 따라) |
| `llm` | 로컬 LLM 기반 쿼리 플래닝 → 병렬 검색 → 종합(실험적) |

<a id="knowledge-library"></a>

### Knowledge Library

```powershell
$env:PYTHONPATH="src"
python -m investigation_search library list
python -m investigation_search library show <session_id>
python -m investigation_search library export --out-dir artifacts/library_export --format html
python -m investigation_search library serve --dir artifacts/library_export --port 8765
python -m investigation_search library publish --out artifacts/library_bundle.zip --format zip
```

<details>
  <summary><b>주의: 삭제 명령</b></summary>

```powershell
python -m investigation_search library delete --yes
```

</details>

<a id="config"></a>

## 설정

<a id="env-vars"></a>

### 환경 변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `PYTHONPATH` | (없음) | `src`를 포함해야 `investigation_search`를 import 가능 |
| `INVESTIGATION_SEARCH_AUTO_INSTALL` | `0` | `1/true/yes/on`이면 선택 의존성을 자동 설치 |
| `INVESTIGATION_USE_PLAYWRIGHT_FETCH` | `0` | `1`이면 웹 fetch에 Playwright(+stealth)를 사용 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama 서버 주소 |

<a id="web-sandbox"></a>

### 웹 검색/격리

- 기본 웹 검색은 로컬 SearxNG(`http://127.0.0.1:8080/search`)를 사용합니다.
- 웹 검색은 기본적으로 **서브프로세스 격리**로 실행됩니다. 제한된 환경에서는 `--no-web-sandbox`가 필요할 수 있습니다.

<details>
  <summary><b>무료 OSINT 수집망(유료 API 대체) 참고</b></summary>

- 기본 웹 검색: 로컬 SearxNG (`http://127.0.0.1:8080/search`)
- 과거 데이터: Wayback Machine CDX API
- 보안/서브도메인 조사: crt.sh / AlienVault OTX (확장용)
- 차단 우회 fetch: Playwright(+stealth) 옵션 (`INVESTIGATION_USE_PLAYWRIGHT_FETCH=1`)
- LLM 모드: Ollama 기반 쿼리 플래너 → 병렬 검색 → 근거 종합 답변 생성

</details>

<a id="development"></a>

## 개발

```powershell
$env:PYTHONPATH="src"
python -m compileall src
```

테스트는 `pytest`가 필요합니다(선택):

```powershell
pip install pytest
pytest -q
```

<a id="contributing"></a>

## 기여

- 이슈/PR 환영합니다. 재현 가능한 로그, 최소 재현 예제, 기대 동작을 함께 남겨주세요.
- 변경 전후로 `python -m compileall src`가 통과하는지 확인해 주세요.
- 자세한 절차: `CONTRIBUTING.md`
- 커뮤니티 규칙: `CODE_OF_CONDUCT.md`
- 보안 취약점 신고: `SECURITY.md`
- 사용 지원: `SUPPORT.md`

<a id="troubleshooting"></a>

## 트러블슈팅

- `TROUBLESHOOTING.md`를 참고하세요.

<a id="changelog"></a>

## 버전/변경 이력

- 버전 단일 소스: `VERSION`
- 변경 이력: `CHANGELOG.md`

<a id="license"></a>

## 라이선스

MIT License. 자세한 내용은 `LICENSE`를 참고하세요.

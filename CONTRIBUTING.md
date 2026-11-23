# Contributing

이 프로젝트에 기여해 주셔서 감사합니다.

## 범위

- 이 저장소는 **온라인 웹 검색 기반** 조사형 검색 엔진을 다룹니다.
- 변경은 가능하면 `schema -> retrieval -> engine` 경계를 유지해 주세요(세부 규칙: `agent.md`).

## 개발 환경

- Python 3.11+
- PowerShell 기준 예시

## 설치(개발용)

```powershell
pip install -r requirements-core.txt
pip install -r requirements-search.txt
pip install -r requirements-tui.txt
pip install -r requirements-docparse.txt
```

필요 시:

```powershell
pip install -r requirements-publish.txt
```

## 실행

```powershell
$env:PYTHONPATH="src"
python -m investigation_search tui --help
```

## 검증(필수)

PR 전 최소한 아래를 통과해 주세요.

```powershell
$env:PYTHONPATH="src"
python -m compileall src
```

테스트(선택):

```powershell
pip install pytest
pytest -q
```

## PR 가이드

- 변경 이유/의도, 사용자가 체감하는 영향(UX/성능/정확도)을 PR 설명에 포함해 주세요.
- 검색 결과 구조(`SourceCitation`, diagnostics) 일관성을 유지해 주세요.
- 새 옵션/환경 변수 추가 시 `README.md`에 함께 문서화해 주세요.


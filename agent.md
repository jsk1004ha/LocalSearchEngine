# agent.md

## 목적
이 저장소는 "로컬 PC에서 동작 가능한 조사형 검색 엔진"의 구현/개선을 목표로 합니다.

## 작업 원칙
1. 검색 결과는 `결론 + 근거 + 반례 + 진단`을 함께 제공한다.
2. 오프라인 빌드는 결정성(재현성)을 보장한다.
3. 온라인 질의는 시간 예산을 준수하고, 생략된 단계는 diagnostics에 기록한다.
4. 로컬 LLM 연동 실패 시 검색 자체는 heuristic 경로로 폴백한다.

## 코드 규칙
- Python 3.11+ 기준 작성.
- 새 기능은 `schema -> retrieval -> engine` 경계를 유지한다.
- 네트워크 종속 기능(LLM)은 예외 발생 시 graceful fallback을 제공한다.

## 검증 규칙
- `python -m compileall src` 성공 확인.
- `PYTHONPATH=src` 스모크 테스트에서 search 결과의 핵심 필드(answer/diagnostics/evidence)가 채워지는지 확인.

## 문서 규칙
- 사용자 가이드는 `README.md`에 유지.
- 버전 변경은 `VERSION` 파일을 단일 소스로 갱신.

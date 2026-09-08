# 데이터 준비와 공개 범위

## 필요한 자료

AWS 관측자료는 [기상청 API허브](https://apihub.kma.go.kr/), 관측소 정보는 [기상자료개방포털](https://data.kma.go.kr/)에서 준비한다. 다운로드 권한과 이용 조건은 각 제공처에서 확인한다. 이 저장소에는 원자료·관측소 메타데이터·지도 원본·SQLite DB를 재배포하지 않는다.

기본 파일 배치는 다음과 같다.

```text
프로젝트/
├── META_관측지점정보_20260517233143.csv
├── WeatherData_AWS/
│   ├── stations/Station_116/AWS_202402.csv
│   ├── stations/Station_.../AWS_YYYYMM.csv
│   └── interpolation_batch/
└── local_data/loo_stage1/loo_raw.db
```

메타데이터에는 최소한 아래 컬럼과 평가 기간에 해당하는 운영 이력이 필요하다. 현재 정보만 내려받은 파일은 과거 운영 이력을 대체하지 못할 수 있다.

| 컬럼 | 형식 |
|---|---|
| `지점` | 문자열 관측소 번호 |
| `지점명` | 지점 표시 이름 |
| `시작일` / `종료일` | 해당 위치 정보의 유효기간, 종료일 결측 허용 |
| `위도` / `경도` | 숫자 좌표 |
| `노장해발고도(m)` | 숫자 고도, 결측 시 기존 fallback 적용 |

메타데이터는 CP949로 읽는다. 다른 이름으로 저장했다면 `KMA_META_FILE` 환경변수 또는 로컬 설정의 `meta_file`에 파일 위치를 지정한다.

관측 CSV는 `TM`(시각)과 `TA`(기온)가 필요하다. `kma.py`가 저장하는 파일은 `STN`과 다른 기상 항목도 포함한다. 파일명과 폴더명은 `Station_<지점>/AWS_YYYYMM.csv` 규칙을 따른다.

## 경로 설정

`project_paths.py`는 아래 환경변수를 먼저 읽고, 없으면 Git에서 제외된 `local_paths.json`, 마지막으로 프로젝트 상대 기본 경로를 사용한다.

| 환경변수 | `local_paths.json` 키 | 기본값 |
|---|---|---|
| `KMA_WORK_DIR` | `work_dir` | `local_data` |
| `KMA_AWS_DIR` | `aws_dir` | `WeatherData_AWS/stations` |
| `KMA_META_FILE` | `meta_file` | 루트의 `META_관측지점정보_20260517233143.csv` |

기존 외부 디스크 자료를 쓰려면 `local_paths.example.json`을 `local_paths.json`으로 복사해 경로를 바꾼다. 상대경로는 프로젝트 루트를 기준으로 해석한다. 외부 작업 폴더 아래에 `loo_stage1/loo_raw.db`, `excluded_stations.csv`, `station_valid_summary.csv`가 함께 있어야 후속 분석을 실행할 수 있다. 대용량 생산에는 동기화 폴더 밖의 작업 경로를 권장한다.

API 키는 `KMA_API_KEY` 환경변수로만 전달한다. 로컬 경로 설정에는 키를 넣지 않는다.

## 공개한 결과 자료

`docs/results/`에는 전국 집계, 초기 감률 비교, 시간대별 고도 회귀계수, 고도집단별 요약과 여섯 관측소 사례 요약을 담았다. `docs/assets/` 그림은 기존 분석에서 만든 결과다. 파일별 원래 위치와 SHA-256은 [결과 출처 목록](results/provenance.json)에 기록했다. 새 관측·실험 결과를 추가하지 않았다.

## 로컬에만 보관하는 항목

- AWS 원자료, 메타데이터 CSV, SQLite·parquet와 관측별 중간 결과
- 참고논문 PDF, 내부 발표·보고서 DOCX/PPTX/HWP, 논문 초안·검토 메모
- `지도데이터/`, 개인 연구용 `기타/`
- 보고서 전용 생성기와 그 전용 테스트, 문서 생성용 Node 의존성
- `AGENTS.md`, `CLAUDE.md`, `CONTEXT.md`, `PROGRESS.md`, `TODO.md`, `.omc/`, `.superpowers/`의 로컬 작업 기록·백업
- 가상환경, 캐시, 임시 산출물, 로그, 개인 녹음·영상, 모델 가중치, 비밀 설정 파일

`.gitignore`는 루트에서 검토한 파일을 이름별로 허용한다. 새 소스 파일을 추가할 때는 공개 적합성을 확인한 후 허용 목록도 함께 갱신한다. 데이터와 비밀 파일을 강제로 추가하지 않는다.

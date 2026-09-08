# 실행 안내

## 환경 설치

Python 3.12를 기준으로 한다. 프로젝트 루트에서 가상환경을 만든다.

```bash
python -m venv .venv
```

Windows PowerShell에서는 `.\.venv\Scripts\Activate.ps1`, macOS/Linux에서는 `source .venv/bin/activate`로 활성화한다. 활성화가 제한되면 Windows에서 `.\.venv\Scripts\python.exe`를 `python` 대신 직접 사용할 수 있다.

```bash
python -m pip install -r requirements.txt
```

기본 의존성은 pandas·NumPy·requests·matplotlib·Pillow·pyarrow다. pyarrow는 전국 생산의 parquet 입출력에 사용한다. 지형 분류·해안 분석·대시보드 자료 준비는 아래 선택 의존성도 설치한다.

```bash
python -m pip install -r requirements-geo.txt
```

기본 패키지 설치와 검증 환경의 세부 사항은 [검증 기록](validation.md)에 적었다. `.ps1` 그림 스크립트는 Windows의 PowerShell·System.Drawing을 사용한다. 한글 그림에는 맑은 고딕 또는 Noto Sans CJK KR 같은 한글 글꼴이 필요하며, 기존 일부 시각화는 맑은 고딕을 기본으로 사용한다.

## 원자료 없이 실행하는 검사

```bash
python test_idw_lapse_methods.py
python test_lapse_rate.py
python test_stage0_invariants.py
python -m unittest test_kma_security test_bias_analysis_core test_bias_data test_bias_geometry test_bias_statistics test_bias_significance test_bias_physical_followup test_bias_physical_block_bootstrap test_elevation_hour_effect test_elevation_hour_standard_metrics
```

이 테스트는 합성 입력으로 계산·짝 비교·집계·경계 조건을 검사한다. 이를 실제 관측 성능으로 해석하지 않는다. 일부 다른 통합 테스트는 로컬 정본 자료나 문헌 map을 요구한다. 전체 suite의 실행·생략 범위는 [검증 기록](validation.md)을 참고한다.

## 데이터 준비와 다운로드

[데이터 안내](data.md)의 메타데이터와 관측 파일을 먼저 준비한다. 직접 다운로드할 때는 운영체제의 환경변수 설정으로 `KMA_API_KEY`를 전달한다. 키를 코드·커밋·공유 로그에 입력하지 않는다. 키가 없으면 실제 HTTP 요청 전에 오류로 중단한다.

`kma.py`는 시작·종료 날짜를 대화형으로 받는다. 연간 자료는 `run_download_2024.py`를 사용하며, 시간당 전체 관측소 요청이므로 백그라운드와 로그를 사용한다. 2024년은 윤년이라 전체 기간 요청은 8,784시간이다.

Windows PowerShell 예시:

```powershell
Start-Process -FilePath (Join-Path $PWD '.venv/Scripts/python.exe') `
  -ArgumentList '-u', 'run_download_2024.py' -WorkingDirectory $PWD `
  -RedirectStandardOutput 'download.log' -RedirectStandardError 'download.err.log' `
  -WindowStyle Hidden
```

macOS/Linux 예시:

```bash
nohup .venv/bin/python -u run_download_2024.py > download.log 2>&1 &
```

기존 다운로드 함수는 **어느 관측소든 해당 월 CSV가 있으면 그 달 전체를 건너뛴다**. 부분 다운로드 뒤의 완전성 검사를 대신하지 않으므로 기간·시간별 누락을 확인해야 한다. 다운로드는 연구자의 API 권한과 네트워크가 필요하며, 이번 문서 정비에서 새로 실행하지 않았다.

## 단일 지점 평가와 초기 다지점 파이프라인

```bash
python run_test.py
```

입력 예시는 시작 `2024-02-01`, 종료 `2024-02-01`, 관측소 `116`이다. 기본 `none` 평가 결과는 `WeatherData_AWS/interpolation_test/idw_result_TA.csv`에 저장된다. 같은 경로의 기존 결과가 있으면 덮어쓰므로 보존이 필요하면 먼저 복사한다.

`run_elevation_compare.py`의 기본 비교는 116번 관측소의 2024년 2월과 4가지 method다. `monthly_lapse`, `band_lapse`를 포함하려면 `build_lapse_tables.py`로 연간 자료에서 감률 테이블을 먼저 준비한다. `none`과 `fixed_lapse`만 비교할 때는 해당 파일의 `METHODS`를 두 방법으로 설정한다.

```bash
python build_lapse_tables.py
python run_elevation_compare.py
python run_pipeline.py
```

`run_pipeline.py`는 초기 다지점 평가 → 집계 → batch 그림 → 이웃 그림 → target 지도로 진행한다. `pipeline_config.py`의 `YEAR`, `TARGET_MODE`, `TARGET_LIST`, `METHODS`를 사용하며 현재 기본 methods는 4가지다. `--only batch`, `--skip batch` 등으로 단계를 선택한다. 지도 단계는 별도 지도 파일이 필요하다. 초기 자동 선정은 9개 target 후보 중 8개만 보간 가능한 기존 구성이다.

## 전체망 생산과 후속 분석

전체망 파이프라인은 초기 다지점 `run_pipeline.py`와 별개다. 큰 작업은 출력 경로를 확인하고 로그를 남겨 실행한다.

| 명령 | 필요한 입력 | 주 출력 |
|---|---|---|
| `python stage0_audit.py` | 메타데이터·2024 AWS | `WeatherData_AWS/interpolation_batch/stage0/` 후보·감사표 |
| `python stage1_produce.py` | 메타데이터·2024 AWS | `KMA_WORK_DIR/loo_stage1/` parquet·SQLite·제외/유효 지점표 |
| `python new_aggregate.py` | SQLite·메타데이터·유효 지점표 | `WeatherData_AWS/interpolation_batch/fullnet/` 전국·계절·월·관측소 집계 |
| `python stage4_regimes.py` | SQLite·유효 지점표 | 이웃 집합의 안정 구간 |
| `python stage5_terrain.py` | 메타데이터·지도·원격 DEM | 관측소 지형 라벨 |
| `python new_query.py` | SQLite·전국 집계표 | 지점·그룹 조회 예시와 그림 |

`stage1_produce.py --merge-only`는 이미 만든 part를 병합한다. SQLite의 조회용 집계표는 `new_query.py`가 필요할 때 생성하므로 해당 DB에는 쓰기 권한도 필요하다. 원자료를 바꾼 뒤 집계표를 재생성하려면 `new_query.build_agg_tables()`를 명시적으로 호출한다.

시간대·고도 분석은 `analyze_elevation_hour_effect.py`, `analyze_elevation_hour_standard_metrics.py`, `analyze_elevation_continuous_metrics.py` 순으로 관련 입력을 준비한다. 기존 BIAS 요인표·시간별 BIAS·coverage·확정 코호트·SQLite 집계가 필요하다. 공개한 요약 CSV만으로 전체 중간 자료를 복원할 수는 없다.

BIAS 후속 보고 runner는 기존 정본 CSV·문헌 map과 검증 manifest도 요구한다. 내부 보고서 자체와 그 전용 생성기는 저장소에 포함하지 않았다. 순수 계산 모듈과 합성 테스트는 원자료 없이 검토할 수 있지만, 내부 보고서 전체의 자동 재생성까지 지원하는 구성은 아니다.

## 지도와 자료 위치

`stage5_terrain.py`는 `지도데이터/korea_boundary_geo.json`과 Copernicus GLO-30 원격 타일을 사용한다. 지도·대시보드 준비에도 경계 자료가 필요하다. 지도 원본과 경량본은 배포하지 않으며 직접 확보한 자료의 좌표계·구조·이용 조건을 확인해야 한다. 원격 DEM 접근은 네트워크가 필요하다.

자료 위치는 환경변수나 `local_paths.json`으로 지정한다. 공개 저장소 기본값은 프로젝트 내 `local_data/`이며, 기존 작업 폴더의 자료를 이동할 필요는 없다.

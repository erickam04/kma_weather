# 분석 구성 안내

기상자료의 준비, 공간 보간, 평가와 오차 진단을 여러 단계로 나누어 사용했습니다. 각 실행 스크립트의 설정과 입력 자료에 따라 분석 범위가 달라집니다.

| 분석 영역 | 내용 | 주요 코드 |
|---|---|---|
| 자료 준비 | API 다운로드, 관측소 운영 이력 선택, 결측·기간 처리 | `weather/kma.py`, `weather/find_stations.py`, `weather/load_aws.py` |
| 공간 보간 | 거리 가중 보간, 이웃 선택과 기여도 추적 | `weather/idw.py`, `weather/fast_idw.py` |
| 고도보정 비교 | 고정·월별·고도구간별 감률의 적용과 비교 | `weather/elevation_correction.py`, `weather/lapse_rate.py`, `scripts/run_*lapse*.py` |
| 다지점·전체망 평가 | 평가 대상을 이웃에서 제외하는 LOO 평가, 병렬 처리, 저장·집계 | `scripts/run_pipeline.py`, `scripts/new_batch_loo.py`, `scripts/stage1_produce.py`, `analysis/new_aggregate.py` |
| 시간·고도별 오차 분석 | 시간대·계절·관측소 고도에 따른 MAE·RMSE·BIAS 비교 | `analysis/analyze_elevation_*.py`, `analysis/analyze_station_*.py` |
| 이웃·지형 요인 분석 | 이웃 배치와 가중 고도차, 해안·섬·산악 특성과 오차의 연관성 | `analysis/bias_geometry.py`, `analysis/h2_direction_test.py`, `analysis/stage5_terrain.py`, `analysis/coastal_threshold_analysis.py` |
| BIAS와 불확실성 분석 | 보정 전후 오차 방향, 짝 비교, 블록 재표집과 유의성 분석 | `analysis/bias_*.py`, `analysis/analyze_bias_*.py` |
| 조회·시각화 | 저장된 집계 조회, 이웃 지도, 시계열·비교 그림, 대시보드 자료 준비 | `analysis/new_query.py`, `visualization/visualize_*.py`, `visualization/plot_*.py`, `visualization/plot_*.ps1`, `visualization/dash_prep.py` |

## 공통 처리 규칙

- 메타데이터 CSV는 CP949, 저장 CSV는 UTF-8 BOM(`utf-8-sig`)을 사용합니다.
- 자료 로드 전의 기간별 후보 선정과 평가 시점의 실제 운영 관측소 선택을 구분합니다.
- 사용자 입력 종료일은 포함하며, 함수 내부의 기간 필터는 `[start, end)`를 사용합니다.
- 고도보정은 거리 가중치를 유지하면서 기온에 적용합니다. 고도가 없을 때는 해당 환산 단계의 원래 값을 사용합니다.
- 오차는 `예측−관측`입니다. MAE는 절대오차, RMSE는 제곱오차, BIAS는 과대·과소추정 방향을 살펴보는 데 사용합니다.
- 관측별 전체 집계와 관측소별 집계는 가중 방식이 다르므로 구분해서 해석합니다.

개별 실험의 수치와 그림은 수록하지 않습니다. 입력 자료와 실행 순서는 [데이터 안내](data.md)와 [실행 안내](running.md)를 참고하세요.

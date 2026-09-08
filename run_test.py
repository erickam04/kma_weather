import os

import pandas as pd

from find_stations import load_station_meta
from idw import TARGET_COLUMN, evaluate_idw, print_score
from load_aws import prepare_evaluation_data


def parse_target_stations(target_input):
    target_input = target_input.strip()

    if target_input == "":
        return None

    return [x.strip() for x in target_input.split(",") if x.strip() != ""]


def main():
    start_date = input("시작 날짜 (예: 2022-01-01): ")
    end_date = input("종료 날짜 (예: 2022-01-31): ")
    target_input = input("테스트할 관측소 번호들 (Ex: 332, 334, 554), 전체는 Enter: ")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    target_stations = parse_target_stations(target_input)

    meta = load_station_meta()
    data, evaluation_meta = prepare_evaluation_data(
        meta,
        start,
        end,
        target_column=TARGET_COLUMN,
    )

    print()
    print("사용 가능한 관측소 수:", len(data.columns))
    print("평가 기간 row 수:", len(data))
    print("보간 대상 변수:", TARGET_COLUMN)
    print("메타데이터 기준: 평가 시각별 자동 갱신")

    result = evaluate_idw(
        data,
        evaluation_meta,
        target_stations=target_stations,
    )
    print_score(result)

    os.makedirs("WeatherData_AWS/interpolation_test", exist_ok=True)
    save_path = f"WeatherData_AWS/interpolation_test/idw_result_{TARGET_COLUMN}.csv"
    result.to_csv(save_path, index=False, encoding="utf-8-sig")

    print()
    print("Saved:", save_path)


if __name__ == "__main__":
    main()


#station 선발 과정이랑 보간과정 시각화해주는 코드
#idw baseline으로 먼저 구조 완성하고 나중에 다른 보간법 구현
#참값 -> 시계열 데이터, x축은 시간, y축은 기온으로 해서 참값과 예측값 plotting 해보기
#시간별로 비교 가능

#  인자는 일단은 기온을 기준으로, 나중에 다른 인자들로 확장할수있게
# idw는 주변 기온을 바탕으로 weighting하는건데, 고도가 다 다름
# 선형적인 고도보정 모델 사용여부 옵션으로 구현
# A지점 test, 해발고도 200m. B랑 C를 보간. B는 50m, C는 300m. B, C의 높이를 0m 기준으로 보정해준 다음 idw 적용. -> 다시 원래
# 고도 기준으로 보정
# 참값 (원래 고도 기준)과 비교 가능
# 고도 보정방식에 대한 조사 (논문) 일단은 기온만

# 1. 시각화
# 2. 참값, 예측값 시간에 대해 plotting


# 3. 고도보정 (자세한 방식은 조사) 기본 방식은 선형으로, 더 자세하게는 조사해보기
# 4. 22년 1월부터 12월까지 더 긴 기간으로, 더 다양한 지점에서 테스트해보고 정리해서 보고

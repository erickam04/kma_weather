import os
import pandas as pd
from project_paths import AWS_DIR

base_folder = str(AWS_DIR)

start_date = input("시작 날짜 (예: 2022-01-01): ")
end_date = input("종료 날짜 (예: 2022-01-31): ")
start = pd.to_datetime(start_date)
end = pd.to_datetime(end_date) + pd.Timedelta(days=1)

#시작일 00시부터 종료일 23시까지 이론상 시간 목록
expected = pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h")
expected_count = len(expected)

#월 목록
months = expected.strftime("%Y%m").unique()
#expected 리스트에서 month들만 뽑음

print("이론상 날짜 수:", expected_count)

problems = 0

for station in os.listdir(base_folder):
    path = os.path.join(base_folder, station)

    if not os.path.isdir(path):
        continue

    data_list = []

    for month in months:
        file_path = os.path.join(path, f"AWS_{month}.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data_list.append(df)

    if len(data_list) == 0:
        print(station, "데이터 없음")
        problems += 1
        continue

    data = pd.concat(data_list, ignore_index=True)
    data["TM"] = pd.to_datetime(data["TM"])

    # 입력한 날짜 범위만 선택
    data = data[(data["TM"] >= start) & (data["TM"] < end)]

    actual_count = len(data)

    saved_times = set(data["TM"])
    missing_times = sorted(set(expected) - saved_times)

    if actual_count != expected_count or len(missing_times) > 0:
        problems += 1

        print()
        print(station, "문제 있음")
        print("실제 row 개수:", actual_count)
        print("이론상 row 개수:", expected_count)

        if len(missing_times) > 0:
            print("빠진 시간:")
            for t in missing_times:
                print(" ", t)


if problems == 0:
    print("모든 station 정상")
else:
    print()
    print("문제 있는 station 수:", problems)

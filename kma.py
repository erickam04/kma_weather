
from project_paths import AWS_DIR
import glob
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests


URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
# API 키는 환경변수로만 전달한다.
API_KEY = os.environ.get("KMA_API_KEY")
BASE_FOLDER = str(AWS_DIR)

COLUMNS = [
    "TM", "STN", "WD1", "WS1", "WDS", "WSS", "WD10", "WS10",
    "TA", "RE", "RN_15m", "RN_60m", "RN_12H", "RN_DAY",
    "HM", "PA", "PS", "TD",
]

VALUE_COLUMNS = [
    "WD1", "WS1", "WDS", "WSS", "WD10", "WS10",
    "TA", "RE", "RN_15m", "RN_60m", "RN_12H", "RN_DAY",
    "HM", "PA", "PS", "TD",
]


def request_data(tm, authkey):
    if not authkey:
        raise ValueError("KMA_API_KEY 환경변수를 설정하세요.")
    params = {
        "tm2": tm,   # 요청 시각
        "stn": 0,    # 전체 station
        "disp": 1,
        "help": 2,
        "authKey": authkey,
    }
    response = requests.get(URL, params=params, timeout=30)
    return response.text


def request_data_with_retry(tm, authkey, retries=2, backoff=3):
    """일시적 실패에 대해 재시도. 최종 실패 시 None 반환."""
    for attempt in range(retries + 1):
        try:
            text = request_data(tm, authkey)
            if text and text.strip():
                return text
        except requests.RequestException as exc:
            print(f"  request error @ {tm} (attempt {attempt + 1}): {type(exc).__name__}")

        if attempt < retries:
            time.sleep(backoff)

    return None


def text_to_dataframe(text):
    rows = []

    for line in text.splitlines():
        line = line.strip()

        if line == "" or line.startswith("#"):
            continue

        data = line.split(",")

        if not data[0].isdigit():
            continue

        if len(data) >= len(COLUMNS):
            rows.append(data[: len(COLUMNS)])

    return pd.DataFrame(rows, columns=COLUMNS)


def month_range(start, end):
    """[start, end) 기간에 포함된 (year, month) 목록."""
    months = []
    cursor = datetime(start.year, start.month, 1)
    while cursor < end:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)
    return months


def month_bounds(year, month, period_start, period_end):
    """해당 월의 실제 다운로드 범위를 전체 기간과 교차시켜 반환."""
    m_start = datetime(year, month, 1)
    if month == 12:
        m_end = datetime(year + 1, 1, 1)
    else:
        m_end = datetime(year, month + 1, 1)

    return max(m_start, period_start), min(m_end, period_end)


def month_already_downloaded(year_month, base_folder=BASE_FOLDER):
    """어느 station이든 해당 월 CSV가 존재하면 이미 받은 것으로 간주."""
    pattern = os.path.join(base_folder, "Station_*", f"AWS_{year_month}.csv")
    return len(glob.glob(pattern)) > 0


def save_month(result, year_month, base_folder=BASE_FOLDER):
    saved = 0
    for station_id, station_data in result.groupby("STN"):
        folder_path = os.path.join(base_folder, f"Station_{station_id}")
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"AWS_{year_month}.csv")
        station_data.to_csv(save_path, index=False, encoding="utf-8-sig")
        saved += 1
    return saved


def download_month(year, month, period_start, period_end, base_folder=BASE_FOLDER, authkey=API_KEY):
    """한 달치를 시간별로 받아 메모리에 모았다가 저장하고 해제한다."""
    year_month = f"{year:04d}{month:02d}"

    if month_already_downloaded(year_month, base_folder):
        print(f"[skip] {year_month} 이미 존재")
        return 0

    m_start, m_end = month_bounds(year, month, period_start, period_end)
    print(f"[month] {year_month} 다운로드 시작 ({m_start} ~ {m_end})")

    data = []
    now = m_start
    while now < m_end:
        tm = now.strftime("%Y%m%d%H%M")
        text = request_data_with_retry(tm, authkey)

        if text is not None:
            df = text_to_dataframe(text)
            if len(df) > 0:
                df["TM"] = pd.to_datetime(df["TM"], format="%Y%m%d%H%M")
                for col in VALUE_COLUMNS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    df.loc[df[col] <= -50, col] = pd.NA
                data.append(df)
        else:
            print(f"  [warn] {tm} 응답 없음, 건너뜀")

        now = now + timedelta(hours=1)

    if len(data) == 0:
        print(f"[month] {year_month} 수집 데이터 없음")
        return 0

    result = pd.concat(data, ignore_index=True)
    saved = save_month(result, year_month, base_folder)
    print(f"[month] {year_month} 저장 완료: station {saved}개")

    # 메모리 해제
    del data, result
    return saved


def download_period(start, end, base_folder=BASE_FOLDER, authkey=API_KEY):
    """[start, end) 기간을 월 단위 증분 저장 방식으로 다운로드한다."""
    for year, month in month_range(start, end):
        download_month(year, month, start, end, base_folder=base_folder, authkey=authkey)


def main():
    start_date = input("시작 날짜 (예: 2024-01-01): ")
    end_date = input("종료 날짜 (예: 2024-12-31): ")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    download_period(start, end)
    print("전체 다운로드 완료")


if __name__ == "__main__":
    main()

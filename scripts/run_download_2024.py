from datetime import datetime

from weather import kma


# 2024년 전체 (2024-01-01 00:00 ~ 2024-12-31 23:00)
# 이미 받은 월(2024-02 등)은 자동 skip, 월 단위 증분 저장.
START = datetime(2024, 1, 1)
END = datetime(2025, 1, 1)


if __name__ == "__main__":
    print("2024 전체 다운로드 시작:", START, "~", END)
    kma.download_period(START, END)
    print("2024 전체 다운로드 완료")

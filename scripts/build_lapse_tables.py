"""2024 전체 AWS 자료로 유동적 lapse rate 테이블(monthly, band)을 생성한다.

시각별 전국 (고도,기온) 회귀로 γ(t)를 추정하고(batch target 제외),
월별 중앙값 테이블과 고도구간 프로파일을 저장한다.

출력:
  WeatherData_AWS/interpolation_batch/monthly_lapse_table.csv
  WeatherData_AWS/interpolation_batch/band_lapse_table.csv
  WeatherData_AWS/interpolation_batch/hourly_gamma_2024.csv  (진단용 시각별 γ)
  WeatherData_AWS/interpolation_visualization/variable_lapse/lapse_tables.png

실행: python -m scripts.build_lapse_tables
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from weather.find_stations import load_station_meta
from weather.idw import TARGET_COLUMN
from weather.lapse_rate import (
    BAND_TABLE_PATH,
    HOURLY_GAMMA_PATH,
    MONTHLY_TABLE_PATH,
    build_band_table,
    build_monthly_table,
    estimate_gamma_series,
    save_table,
)
from weather.load_aws import prepare_evaluation_data
from scripts.run_batch_test import discover_months, pick_target_stations
import pipeline_config as cfg

VIZ = os.path.join(cfg.viz_dir("variable_lapse"), "lapse_tables.png")


def setup_korean_font():
    for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def main():
    meta = load_station_meta()
    # γ 추정 회귀에서 제외할 batch target (in-sample 완화)
    exclude = pick_target_stations(meta)["지점"].astype(str).tolist()
    print("γ 회귀에서 제외할 target:", exclude)

    months = discover_months(2024)
    print("월:", months)

    all_gamma, all_band = [], []
    for ym in months:
        y, m = int(ym[:4]), int(ym[4:])
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthBegin(1)
        print(f"  {ym} 로드·회귀 중 ...")
        data, emeta = prepare_evaluation_data(meta, start, end, target_column=TARGET_COLUMN)
        g, b = estimate_gamma_series(data, emeta, exclude_stations=exclude)
        all_gamma.append(g)
        all_band.append(b)

    gamma_df = pd.concat(all_gamma, ignore_index=True)
    band_df = pd.concat(all_band, ignore_index=True)
    gamma_df.to_csv(HOURLY_GAMMA_PATH, index=False, encoding="utf-8-sig")

    monthly = build_monthly_table(gamma_df)
    band = build_band_table(band_df)
    save_table(monthly, MONTHLY_TABLE_PATH)
    save_table(band, BAND_TABLE_PATH)

    print("\n=== 월별 lapse rate 테이블 (°C/km) ===")
    print(monthly.assign(gamma_C_per_km=(monthly["gamma"] * 1000).round(2))
          .to_string(index=False))
    print("\n=== 고도구간 lapse rate 테이블 (°C/km) ===")
    print(band.assign(gamma_C_per_km=(band["gamma"] * 1000).round(2))
          .to_string(index=False))

    # 시각화
    setup_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(monthly["month"], monthly["gamma"] * 1000, "o-", color="#2d6cdf")
    ax.axhline(-6.5, ls="--", color="#888", label="고정 -6.5")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("월"); ax.set_ylabel("lapse rate (°C/km)")
    ax.set_title("월별 lapse rate (2024, 전국 회귀 중앙값)", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    edges = [0.0] + [r for r in band["z_hi"] if r != float("inf")]
    edges = edges + [max(edges) + 500 if edges else 2000]  # inf 구간 시각화용 상한
    for i, (_, r) in enumerate(band.iterrows()):
        lo = r["z_lo"]
        hi = r["z_hi"] if r["z_hi"] != float("inf") else edges[-1]
        ax.plot([lo, hi], [r["gamma"] * 1000] * 2, "-", lw=3, color="#d1913c")
    ax.axhline(-6.5, ls="--", color="#888", label="고정 -6.5")
    ax.set_xlabel("고도 (m)"); ax.set_ylabel("lapse rate (°C/km)")
    ax.set_title("고도구간별 lapse rate (2024)", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(VIZ), exist_ok=True)
    fig.savefig(VIZ, dpi=150)
    plt.close(fig)
    print(f"\n저장: {MONTHLY_TABLE_PATH}\n     {BAND_TABLE_PATH}\n     {VIZ}")


if __name__ == "__main__":
    main()

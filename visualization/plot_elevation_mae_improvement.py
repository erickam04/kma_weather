from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path('WeatherData_AWS/interpolation_batch/fullnet/elevation_hour_effect')
DATA = BASE / 'standard_metrics_station_hour.csv'
COHORT = BASE / 'station_cohort.csv'
OUT = BASE / 'plots' / '10_elevation_vs_mae_improvement.png'


def main():
    metrics = pd.read_csv(DATA)
    cohort = pd.read_csv(COHORT)
    cohort = cohort[cohort['COHORT_MAIN'].astype(str).str.lower().eq('true')][['STN']]
    metrics = metrics.merge(cohort, on='STN', how='inner')
    wide = (
        metrics.pivot_table(index=['STN', '지점명', 'elev_m', 'HOUR'],
                            columns='METHOD', values='MAE_C', aggfunc='first')
        .reset_index()
    )
    wide = wide.dropna(subset=['none', 'fixed_lapse', 'elev_m'])
    wide['improvement_C'] = wide['none'] - wide['fixed_lapse']

    def fit(hour):
        d = wide[wide['HOUR'].eq(hour)]
        x = d['elev_m'].to_numpy(float)
        y = d['improvement_C'].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        return d, slope, intercept

    hourly = []
    for hour in sorted(wide['HOUR'].unique()):
        d, slope, intercept = fit(hour)
        hourly.append((hour, slope * 100, len(d)))
    hourly = pd.DataFrame(hourly, columns=['hour', 'slope_C_per_100m', 'n'])
    hourly.to_csv(BASE / 'elevation_vs_mae_improvement_hourly.csv', index=False, encoding='utf-8-sig')

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 8.2), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.9], hspace=0.34, wspace=0.25)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    colors = {'positive': '#2b8cbe', 'negative': '#d95f0e'}
    for ax, hour in zip(axes, [6, 15]):
        d, slope, intercept = fit(hour)
        pos = d['improvement_C'] >= 0
        ax.scatter(d.loc[pos, 'elev_m'], d.loc[pos, 'improvement_C'], s=13,
                   alpha=0.58, color=colors['positive'], label='개선(양수)')
        ax.scatter(d.loc[~pos, 'elev_m'], d.loc[~pos, 'improvement_C'], s=13,
                   alpha=0.58, color=colors['negative'], label='악화(음수)')
        xx = np.linspace(d['elev_m'].min(), d['elev_m'].max(), 100)
        ax.plot(xx, slope * xx + intercept, color='black', lw=1.5,
                label=f'회귀선: {slope*100:+.3f} °C/100 m')
        ax.axhline(0, color='gray', lw=0.9)
        ax.set_title(f'{hour:02d}시', fontsize=13, weight='bold')
        ax.set_xlabel('관측소 해발고도 (m)')
        ax.set_ylabel('MAE 개선량 (보정 전 − 보정 후, °C)')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc='best', frameon=True)
        ax.text(0.03, 0.95, f'n={len(d)}', transform=ax.transAxes, va='top', fontsize=9)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(hourly['hour'], hourly['slope_C_per_100m'], marker='o', color='#6a51a3', lw=2, ms=4)
    ax.axhline(0, color='black', lw=0.9)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(24))
    ax.set_xlabel('시각 (KST)')
    ax.set_ylabel('고도 100 m 증가 시 MAE 개선량 변화 (°C)')
    ax.set_title('시간대별: 고도가 높아질수록 고도보정 개선량이 얼마나 달라지는가', fontsize=13, weight='bold')
    ax.grid(alpha=0.2)
    fig.suptitle('관측소 고도와 고도보정에 따른 MAE 개선량의 관계', fontsize=16, weight='bold', y=0.98)
    fig.text(0.5, 0.01, '양수: 고도보정 후 MAE 감소(개선)  |  음수: MAE 증가(악화)  |  회귀기울기는 평균적인 경향',
             ha='center', fontsize=9)
    fig.savefig(OUT, bbox_inches='tight')
    print(OUT)
    print(hourly.to_string(index=False))


if __name__ == '__main__':
    main()

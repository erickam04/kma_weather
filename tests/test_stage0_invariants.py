"""Stage 0 엣지케이스 불변식 — 합성 단위테스트 (계획서 §10 Stage 0 ③).

실데이터(2024 AWS)에 인스턴스가 0개인 엣지케이스는 합성 데이터로만 검증 가능:
  - NaN 고도 fallback (감사 결과 실데이터 0개)
  - 좌표중복 distance==0 분기 (감사 결과 실데이터 0개)
그 외 tie·LOO self-index·none 불변식도 함께 확정.

알고리즘 무수정 — idw / elevation_correction 함수를 합성 입력으로 직접 호출.
실행: python -m tests.test_stage0_invariants  (pytest 불요, assert 기반)
"""

import pandas as pd

from weather.idw import idw_interpolate_with_trace
from weather.elevation_correction import FIXED_LAPSE_RATE

GAMMA = FIXED_LAPSE_RATE  # -0.0065


def make_meta(rows):
    """rows: list of (지점, 위도, 경도, 고도). 지점명은 자동."""
    return pd.DataFrame([
        {"지점": str(s), "지점명": f"S{s}", "위도": lat, "경도": lon,
         "노장해발고도(m)": elev}
        for s, lat, lon, elev in rows
    ])


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


results = []
def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"  [{'OK' if cond else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------------------
# 기본 세팅: target 500m, 저지대 이웃 3개(100m, 모두 10.0도)
# ---------------------------------------------------------------------------
def base_case(elevation_method):
    meta = make_meta([
        ("T", 37.00, 127.00, 500.0),
        ("N1", 37.02, 127.00, 100.0),
        ("N2", 37.00, 127.02, 100.0),
        ("N3", 36.98, 127.00, 100.0),
    ])
    values = pd.Series({"N1": 10.0, "N2": 10.0, "N3": 10.0})
    return idw_interpolate_with_trace("T", values, meta,
                                      elevation_method=elevation_method)


print("=== 1. none 불변식 (보정 없음) ===")
pred, tr = base_case("none")
# 이웃 전부 10.0 → 예측 10.0, 고도조정 없음
check("none: 예측=이웃평균", approx(pred, 10.0), f"pred={pred}")
check("none: elevation_adjustment 없음(NA)", tr["elevation_adjustment"].isna().all()
      or (tr["value"] == tr["raw_value"]).all(), "value==raw_value")

print("=== 2. fixed_lapse 결정값 ===")
pred, tr = base_case("fixed_lapse")
# 이웃 10.0 -> 기준면 10.0+(-0.0065)(0-100)=10.65 -> IDW 10.65
# -> target 500m: 10.65+(-0.0065)(500-0)=7.4
check("fixed_lapse: 예측=7.4", approx(pred, 7.4, 1e-9), f"pred={pred}")

print("=== 3. NaN 고도 fallback (이웃) ===")
meta = make_meta([
    ("T", 37.00, 127.00, 500.0),
    ("N1", 37.02, 127.00, float("nan")),  # 고도 결측
    ("N2", 37.00, 127.02, 100.0),
    ("N3", 36.98, 127.00, 100.0),
])
values = pd.Series({"N1": 10.0, "N2": 10.0, "N3": 10.0})
pred, tr = idw_interpolate_with_trace("T", values, meta, elevation_method="fixed_lapse")
n1 = tr[tr["station_id"] == "N1"].iloc[0]
# N1: 고도 NaN -> 기준면 환산 없이 원본 10.0 그대로
check("NaN이웃: value_at_reference=원본(10.0)", approx(n1["value_at_reference"], 10.0),
      f"={n1['value_at_reference']}")
n2 = tr[tr["station_id"] == "N2"].iloc[0]
check("정상이웃 N2: 기준면 10.65", approx(n2["value_at_reference"], 10.65),
      f"={n2['value_at_reference']}")

print("=== 4. NaN 고도 fallback (target) ===")
meta = make_meta([
    ("T", 37.00, 127.00, float("nan")),   # target 고도 결측
    ("N1", 37.02, 127.00, 100.0),
    ("N2", 37.00, 127.02, 100.0),
    ("N3", 36.98, 127.00, 100.0),
])
values = pd.Series({"N1": 10.0, "N2": 10.0, "N3": 10.0})
pred, tr = idw_interpolate_with_trace("T", values, meta, elevation_method="fixed_lapse")
# 이웃 기준면 10.65, target 고도 NaN -> from_reference fallback -> 10.65 그대로
check("NaN target: 예측=기준면값(10.65)", approx(pred, 10.65), f"pred={pred}")

print("=== 5. 좌표중복 distance==0 분기 ===")
meta = make_meta([
    ("T", 37.00, 127.00, 500.0),
    ("N0", 37.00, 127.00, 500.0),  # target과 완전 동일 좌표·고도
    ("N1", 37.02, 127.00, 100.0),
    ("N2", 37.00, 127.02, 100.0),
])
values = pd.Series({"N0": 15.0, "N1": 10.0, "N2": 10.0})
pred, tr = idw_interpolate_with_trace("T", values, meta, elevation_method="fixed_lapse")
# distance==0 이웃 N0가 weight=inf로 단독 채택. N0 고도=target 고도 -> 왕복 상쇄 -> 15.0
check("dist0: 예측=중복이웃값(15.0)", approx(pred, 15.0), f"pred={pred}")
n0 = tr[tr["station_id"] == "N0"].iloc[0]
check("dist0: weight_normalized=1.0", approx(n0["weight_normalized"], 1.0),
      f"={n0['weight_normalized']}")
pred_none, _ = idw_interpolate_with_trace("T", values, meta, elevation_method="none")
check("dist0(none): 예측=중복이웃 원본(15.0)", approx(pred_none, 15.0), f"pred={pred_none}")

print("=== 6. max_neighbors tie (동일거리 11개 중 10개 채택) ===")
rows = [("T", 37.00, 127.00, 100.0)]
for i in range(11):
    rows.append((f"M{i}", 37.10, 127.00, 100.0))  # 전부 동일 좌표=동일 거리
meta = make_meta(rows)
values = pd.Series({f"M{i}": float(i) for i in range(11)})
pred, tr = idw_interpolate_with_trace("T", values, meta, elevation_method="none")
check("tie: 정확히 max_neighbors(10)개 채택", len(tr) == 10, f"len={len(tr)}")
# sorted 안정 -> 삽입순(M0..M9) 유지, M10 탈락
kept = set(tr["station_id"])
check("tie: 삽입순 앞 10개(M0..M9) 유지, M10 제외",
      kept == {f"M{i}" for i in range(10)}, f"M10 in={('M10' in kept)}")

print("=== 7. LOO self-index (target 자기값 미사용) ===")
meta = make_meta([
    ("T", 37.00, 127.00, 100.0),
    ("N1", 37.02, 127.00, 100.0),
    ("N2", 37.00, 127.02, 100.0),
    ("N3", 36.98, 127.00, 100.0),
])
values = pd.Series({"T": 999.0, "N1": 10.0, "N2": 10.0, "N3": 10.0})  # 자기값 오염
pred, tr = idw_interpolate_with_trace("T", values, meta, elevation_method="none")
check("self: target 이웃목록에 없음", "T" not in set(tr["station_id"]), "")
check("self: 예측이 자기값(999)에 오염 안 됨", approx(pred, 10.0), f"pred={pred}")

print()
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{'='*50}")
print(f"결과: {len(results)-n_fail}/{len(results)} 통과",
      "-> [PASS]" if n_fail == 0 else f"-> [FAIL] {n_fail}건 실패")

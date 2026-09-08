import pandas as pd


FIXED_LAPSE_RATE = -0.0065  # degC/m (고도 100m 상승 시 약 0.65도 하강)
REFERENCE_ELEVATION_M = 0.0  # 기준면 = 해수면(0m)


def to_reference_elevation(
    value,
    source_elevation_m,
    method="none",
    lapse_rate=FIXED_LAPSE_RATE,
    reference_elevation_m=REFERENCE_ELEVATION_M,
):
    """이웃 기온을 원래 고도(source)에서 기준면(0m) 값으로 환산한다.

    T_ref = T_source + lapse_rate * (z_ref - z_source)

    이렇게 모든 이웃을 같은 기준면으로 맞춘 뒤 IDW를 적용한다.
    method="none"이거나 고도값이 결측이면 원래 값을 그대로 반환한다.
    """
    if method == "none":
        return value

    if pd.isna(value) or pd.isna(source_elevation_m):
        return value

    if method == "fixed_lapse":
        return value + lapse_rate * (reference_elevation_m - source_elevation_m)

    raise ValueError(f"unknown elevation_method: {method}")


def from_reference_elevation(
    value,
    target_elevation_m,
    method="none",
    lapse_rate=FIXED_LAPSE_RATE,
    reference_elevation_m=REFERENCE_ELEVATION_M,
):
    """기준면(0m)에서 보간된 값을 target station 고도로 되돌린다.

    T_target = T_ref + lapse_rate * (z_target - z_ref)

    IDW 보간이 끝난 뒤 예측값 한 번에 적용한다.
    method="none"이거나 target 고도가 결측이면 원래 값을 그대로 반환한다.
    """
    if method == "none":
        return value

    if pd.isna(value) or pd.isna(target_elevation_m):
        return value

    if method == "fixed_lapse":
        return value + lapse_rate * (target_elevation_m - reference_elevation_m)

    raise ValueError(f"unknown elevation_method: {method}")

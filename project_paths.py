"""자료와 시스템 글꼴의 위치를 한곳에서 관리한다.

환경변수가 로컬 설정보다 우선한다. local_paths.json은 Git에서 제외하며
기존 외부 자료를 이동하지 않고 사용할 때만 작성한다.
"""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
_config_path = ROOT / "local_paths.json"
_local = json.loads(_config_path.read_text(encoding="utf-8-sig")) if _config_path.exists() else {}


def _path(env_name, config_name, default):
    value = Path(os.environ.get(env_name) or _local.get(config_name) or default).expanduser()
    return value if value.is_absolute() else ROOT / value


WORK_DIR = _path("KMA_WORK_DIR", "work_dir", "local_data")
AWS_DIR = _path("KMA_AWS_DIR", "aws_dir", "WeatherData_AWS/stations")
META_PATH = _path("KMA_META_FILE", "meta_file", "META_관측지점정보_20260517233143.csv")
LOO_DIR = WORK_DIR / "loo_stage1"
LOO_DB = LOO_DIR / "loo_raw.db"
WINDOWS_FONT_DIR = Path(os.environ.get("WINDIR", "Windows")) / "Fonts"

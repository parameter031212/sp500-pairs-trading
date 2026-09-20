from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import math
import tomllib

import pandas as pd


class ConfigError(ValueError):
    pass


def merge(base: dict, updates: dict) -> dict:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"설정 파일 상속이 순환합니다: {path}")
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    parent = values.pop("extends", None)
    if parent:
        values = merge(_read(path.parent / parent, seen | {path}), values)
    return values


@dataclass(frozen=True)
class Config:
    values: dict
    project_root: Path
    source: Path

    def __getitem__(self, section: str) -> dict:
        return deepcopy(self.values[section])

    def with_updates(self, updates: dict) -> "Config":
        values = deepcopy(self.values)
        for dotted_key, value in updates.items():
            parts = dotted_key.split(".")
            target = values
            for key in parts[:-1]:
                if key not in target or not isinstance(target[key], dict):
                    raise ConfigError(f"알 수 없는 설정: {dotted_key}")
                target = target[key]
            if parts[-1] not in target:
                raise ConfigError(f"알 수 없는 설정: {dotted_key}")
            target[parts[-1]] = value
        _validate(values)
        return Config(values, self.project_root, self.source)

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        split = self.values["evaluation"]["active_split"]
        section = self.values["schedule"] if split == "all" else self.values["evaluation"]
        prefix = "" if split == "all" else split + "_"
        return pd.Timestamp(section[prefix + "start"]), pd.Timestamp(section[prefix + "end"])


def _validate(v: dict) -> None:
    schema_path = Path(__file__).with_name("defaults.toml")
    if schema_path.exists():
        schema = _read(schema_path, set())
        if set(v) != set(schema):
            raise ConfigError(f"설정 섹션이 다릅니다: {set(v) ^ set(schema)}")
        for section in schema:
            if not isinstance(v[section], dict) or set(v[section]) != set(schema[section]):
                raise ConfigError(f"{section}: 누락되거나 알 수 없는 설정 키")
            for key, default in schema[section].items():
                value = v[section][key]
                if isinstance(default, bool) and not isinstance(value, bool):
                    raise ConfigError(f"{section}.{key}는 true/false여야 합니다")
                if isinstance(default, (int, float)) and not isinstance(default, bool):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ConfigError(f"{section}.{key}는 유한한 숫자여야 합니다")
                    if isinstance(default, int) and not isinstance(value, int):
                        raise ConfigError(f"{section}.{key}는 정수여야 합니다")
                if isinstance(default, str) and not isinstance(value, str):
                    raise ConfigError(f"{section}.{key}는 문자열이어야 합니다")
                if isinstance(default, list) and (not isinstance(value, list) or not all(isinstance(x, str) for x in value)):
                    raise ConfigError(f"{section}.{key}는 문자열 목록이어야 합니다")
    else:
        raise ConfigError("설정 검증용 defaults.toml이 없습니다. 배포 파일을 확인하세요.")
    d, s, f, n, t, p, c, e, r = (v[x] for x in ("data", "schedule", "formation", "selection", "strategy", "portfolio", "costs", "evaluation", "run"))
    def require(condition: bool, message: str):
        if not condition:
            raise ConfigError(message)
    require(d["mode"] in {"csv", "synthetic"}, "data.mode: csv/synthetic")
    require(bool(v["corporate_actions"]["events_file"]) or not v["corporate_actions"]["enabled"], "합병 일정 파일 경로 필요")
    require(0 < d["min_formation_coverage"] <= 1, "커버리지는 0 초과 1 이하")
    require(d["min_formation_observations"] >= 20, "최소 포메이션 관측치는 20 이상")
    require(s["formation_months"] >= 1 and s["trading_months"] >= 1, "개월 수는 1 이상")
    require(f["orientation"] in {"fixed", "best_of_two"}, "orientation: fixed/best_of_two")
    require(f["multiple_testing"] in {"none", "holm", "fdr_by"}, "multiple_testing: none/holm/fdr_by")
    require(f["autolag"] in {"AIC", "BIC", "fixed"}, "autolag: AIC/BIC/fixed")
    require(f["maxlag"] >= -1 and (f["autolag"] != "fixed" or f["maxlag"] >= 0), "fixed 시차는 maxlag를 0 이상으로 지정")
    require(0 < f["cointegration_alpha"] < 1 and 0 < f["i1_alpha"] < 1, "유의수준은 0과 1 사이")
    require(-1 <= f["min_return_correlation"] <= 1, "상관계수 기준은 -1~1")
    require(0 < f["beta_min"] <= f["beta_max"], "양의 beta_min <= beta_max 필요")
    require(0 < f["half_life_min"] <= f["half_life_max"], "양의 반감기 범위 필요")
    require(n["max_pairs"] >= 1, "최대 페어 수는 1 이상")
    require(n["max_per_sector"] >= 0 and n["max_per_stock"] >= 0,
            "집중 제한은 0 이상이며 0은 제한 없음")
    h = v["pretrade"]
    require(h["months"] >= 1 and h["block_months"] >= 1 and h["months"] % h["block_months"] == 0,
            "pretrade.months는 block_months의 양의 배수여야 합니다")
    require(h["min_entries"] >= 1, "사전 모의거래 최소 진입은 1 이상")
    require(h["entry_z"] > 0 and -h["entry_z"] < h["exit_z"] < h["entry_z"] and h["stop_z"] > h["entry_z"],
            "사전 모의거래의 진입/청산/손절 기준 확인")
    require(t["entry_z"] > 0 and -t["entry_z"] < t["exit_z"] < t["entry_z"], "청산 기준은 -entry_z와 entry_z 사이")
    require(t["stop_z"] > 0 and (not t["stop_enabled"] or t["stop_z"] > t["entry_z"]), "활성 손절 기준은 진입 기준보다 커야 함")
    require(t["max_holding_days"] >= 2, "보유기간은 체결일 포함 2거래일 이상")
    require(t["reentry_after_stop"] in {"block_window", "after_reset"} and t["reentry_after_time_exit"] in {"block_window", "after_reset"}, "재진입 정책: block_window/after_reset")
    require(p["initial_capital"] > 0 and p["cash_annual_rate"] > -1, "자본/현금 이율 확인")
    require(all(value >= 0 for value in c.values()), "비용은 음수일 수 없음")
    require(e["active_split"] in {"all", "train", "validation", "test"}, "평가 구간 이름 오류")
    require(e["annualization"] > 0 and e["risk_free_annual_rate"] > -1, "연율화 설정 오류")
    require(r["max_windows"] >= 0 and r["seed"] >= 0, "실행 제한/시드 오류")
    try:
        start, end = pd.Timestamp(s["start"]), pd.Timestamp(s["end"])
        require(not pd.isna(start) and not pd.isna(end) and start <= end, "실험 시작/종료일 확인")
        previous_end = None
        for split in ("train", "validation", "test"):
            a, b = e[split + "_start"], e[split + "_end"]
            require(bool(a) == bool(b), f"{split}의 시작/종료일을 모두 지정")
            if a:
                a, b = pd.Timestamp(a), pd.Timestamp(b)
                require(start <= a <= b <= end, f"{split}은 전체 실험 기간 안에 있어야 함")
                require(previous_end is None or a > previous_end, "학습/검증/시험 기간은 시간순으로 겹치지 않아야 함")
                previous_end = b
        require(e["active_split"] == "all" or bool(e[e["active_split"] + "_start"]), "선택한 평가 구간 날짜가 아직 미정입니다")
    except (ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc


def load_config(path: str | Path = "configs/baseline.toml") -> Config:
    source = Path(path).expanduser().resolve()
    # New optional fields must also work with previously saved full configurations.
    values = merge(_read(Path(__file__).with_name("defaults.toml"), set()), _read(source, set()))
    _validate(values)
    root = source.parent.parent if source.parent.name == "configs" else source.parent
    return Config(values, root, source)

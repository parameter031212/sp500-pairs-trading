"""Ordered, checkpointed A–D × formation × exit × stop experiments."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ConfigError
from .data import DataError, load_data
from .formation import Pair
from .runner import run_backtest, source_fingerprint, write_json
from .schedule import make_windows
from .corporate_actions import calendar_for


MODELS = {
    "A": (False, False, "공적분"),
    "B": (True, False, "공적분 + 상관관계"),
    "C": (False, True, "공적분 + 사전 거래 샤프"),
    "D": (True, True, "공적분 + 상관관계 + 사전 거래 샤프"),
}
FORMATION_MONTHS = (12, 24)
EXIT_Z = (0.0, 0.5)
STOP_Z = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)


def _number(value):
    return f"{value:.1f}".replace("-", "minus").replace(".", "p")


def build_study_plan():
    rows = []
    for model, (_, _, label) in MODELS.items():
        for months in FORMATION_MONTHS:
            for exit_z in EXIT_Z:
                for stop_z in STOP_Z:
                    rows.append({"experiment": len(rows) + 1, "model": model, "model_label": label,
                        "name": f"{model}_F{months}_Exit{_number(exit_z)}_Stop{_number(stop_z)}",
                        "formation_months": months, "exit_z": exit_z, "stop_z": stop_z})
    return pd.DataFrame(rows)


def study_history_requirements(base, calendar, models=None):
    """Report warm-up coverage without changing dates or running any experiments.

    The suggested first window preserves the requested trading-month grid.
    This checks history length only; per-stock coverage is checked during fitting.
    """
    chosen = set(MODELS if models is None else models)
    if not chosen or not chosen <= set(MODELS):
        raise ConfigError("models에는 A/B/C/D 중 하나 이상을 지정하세요")
    calendar = pd.DatetimeIndex(calendar)
    if calendar.empty:
        raise DataError("가격 날짜가 없습니다")
    start, end = base.date_range()
    step = pd.DateOffset(months=base["schedule"]["trading_months"])
    first_window = pd.Timestamp(base["schedule"]["start"])
    while first_window < start:
        first_window += step
    if first_window + step - pd.Timedelta(days=1) > end:
        raise DataError("선택한 날짜 범위에 완전한 거래 구간이 없습니다. 평가 분할을 거래 구간 경계에 맞추세요.")
    data_start = calendar.min()
    tolerance = pd.Timedelta(days=7)  # Same holiday allowance as make_windows/historical_score.
    rows = []
    for model in MODELS:
        if model not in chosen:
            continue
        pretrade_months = base["pretrade"]["months"] if MODELS[model][1] else 0
        for months in FORMATION_MONTHS:
            lookback = pd.DateOffset(months=months + pretrade_months)
            required = first_window - lookback
            suggested = first_window
            while data_start > suggested - lookback + tolerance:
                suggested += step
            rows.append({"model": model, "formation_months": months,
                "pretrade_months": pretrade_months, "history_months": months + pretrade_months,
                "first_window": str(first_window.date()), "required_price_start": str(required.date()),
                "actual_price_start": str(data_start.date()),
                "history_available": bool(data_start <= required + tolerance),
                "suggested_first_window": str(suggested.date())})
    return pd.DataFrame(rows)


def experiment_config(base, row):
    correlation, sharpe, _ = MODELS[row["model"]]
    return base.with_updates({
        "schedule.formation_months": int(row["formation_months"]),
        "formation.correlation_filter": correlation, "pretrade.enabled": sharpe,
        "strategy.entry_z": 2.0, "strategy.exit_z": float(row["exit_z"]),
        "strategy.stop_enabled": True, "strategy.stop_z": float(row["stop_z"]),
    })


def _encode(value):
    if isinstance(value, pd.DataFrame):
        # Python's JSON float representation round-trips exactly, unlike
        # DataFrame.to_json's fixed decimal precision near a filter boundary.
        return {"_kind": "frame", "value": _encode(value.to_dict(orient="split")),
            "dtypes": {k: str(v) for k, v in value.dtypes.items()}, "index_name": value.index.name}
    if isinstance(value, Pair):
        return {"_kind": "pair", "value": _encode(asdict(value))}
    if isinstance(value, pd.Timestamp):
        return {"_kind": "date", "value": str(value)}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"_kind": "float", "value": str(value)}
    if isinstance(value, tuple):
        return {"_kind": "tuple", "value": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        kind = value.get("_kind")
        if kind == "frame":
            frame = pd.DataFrame(**_decode(value["value"])).astype(value["dtypes"])
            frame.index.name = value["index_name"]
            return frame
        if kind == "pair":
            return Pair(**_decode(value["value"]))
        if kind == "date":
            return pd.Timestamp(value["value"])
        if kind == "float":
            return float(value["value"])
        if kind == "tuple":
            return tuple(_decode(v) for v in value["value"])
        return {k: _decode(v) for k, v in value.items()}
    return value


def _atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, value)
    temp.replace(path)


class SelectionCache:
    """Small memory cache backed by JSON; reusable after a process restart."""
    def __init__(self, directory, data_identity, memory_limit=24):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.data_identity = data_identity
        self.memory_limit = memory_limit
        self.memory = OrderedDict()

    def _path(self, key):
        return self.directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def __contains__(self, key):
        return key in self.memory or self._path(key).is_file()

    def _remember(self, key, value):
        self.memory[key] = value
        self.memory.move_to_end(key)
        while len(self.memory) > self.memory_limit:
            self.memory.popitem(last=False)

    def __getitem__(self, key):
        if key not in self.memory:
            payload = json.loads(self._path(key).read_text(encoding="utf-8"))
            if payload["key"] != key:
                raise ConfigError("선정 캐시 식별자가 다릅니다")
            self._remember(key, _decode(payload["value"]))
        self.memory.move_to_end(key)
        return self.memory[key]

    def __setitem__(self, key, value):
        _atomic_json(self._path(key), {"key": key, "value": _encode(value)})
        self._remember(key, value)


def _data_fingerprint(data):
    digest = hashlib.sha256()
    for frame in (data.prices, data.membership[["date", "tickers"]], data.sectors):
        digest.update(json.dumps(list(frame.columns), ensure_ascii=False).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _checkpoint(output, state):
    records = state["experiments"]
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["counts"] = {status: sum(r["status"] == status for r in records)
        for status in ("pending", "running", "complete", "failed", "interrupted")}
    state["status"] = "complete" if state["counts"]["complete"] == len(records) else "incomplete"
    _atomic_json(output / "study_manifest.json", state)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "summary"} | r.get("summary", {}) for r in records])
    for name, frame in [("comparison", table)] + [(f"model_{m}_comparison", table[table["model"] == m]) for m in MODELS]:
        temp = output / (name + ".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(output / (name + ".csv"))
    return table


def _verify_completed(output, record):
    directory = output / record["output_directory"]
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        files = ("effective_config.json", "portfolio_daily.csv", "trades.csv", "windows.csv", "selected_pairs.csv", "merger_events.json")
        return (manifest["status"] == "complete" and summary == record["summary"]
            and all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in files))
    except (OSError, ValueError, KeyError):
        return False


def run_selection_study(base, data=None, data_dir=None, output_dir=None, models=None,
                        resume=True, retry_failed=False, progress=print, max_experiments=None,
                        output_profile="compact"):
    """Run all A trials before B, C, D; checkpoint after every trial.

    `models=["A"]` runs only A without changing the 144-trial study identity.
    Re-run with the same data/config/output to continue. Failed trials are retained
    unless retry_failed=True. A failed trial never contributes performance metrics.
    """
    if base["evaluation"]["active_split"] == "test":
        raise ConfigError("최종 test 기간에서는 파라미터 탐색을 실행할 수 없습니다")
    chosen = set(MODELS if models is None else models)
    if not chosen or not chosen <= set(MODELS):
        raise ConfigError("models에는 A/B/C/D 중 하나 이상을 지정하세요")
    if max_experiments is not None and (isinstance(max_experiments, bool) or not isinstance(max_experiments, int) or max_experiments < 1):
        raise ConfigError("max_experiments는 양의 정수 또는 None")
    if output_profile not in {"compact", "full"}:
        raise ConfigError("output_profile: compact/full")
    plan = build_study_plan()
    configs = [experiment_config(base, r) for r in plan.to_dict("records")]
    data = data if data is not None else load_data(base, data_dir)
    mergers = calendar_for(data, base)
    history = study_history_requirements(base, data.prices.index, chosen)
    if not history["history_available"].all():
        limiting = history.sort_values("history_months", ascending=False).iloc[0]
        label = "샤프 필터 포함" if limiting.pretrade_months else "포메이션"
        raise DataError(
            f"실험 시작 전 가격 자료 부족 ({label}). "
            f"현재 첫 거래 구간={limiting.first_window}, 실제 가격 시작={limiting.actual_price_start}. "
            f"{limiting.model}의 추정 {limiting.formation_months}개월 + 사전 모의거래 "
            f"{limiting.pretrade_months}개월 = 총 {limiting.history_months}개월이 필요하므로 "
            f"{limiting.required_price_start}부터 가격이 있어야 합니다. "
            f"선택한 모델을 같은 기간으로 비교하려면 현재 거래 주기에서 "
            f"{history.suggested_first_window.max()} 이후 구간을 사용하세요. "
            "설정 셀에서 시작일과 필요시 종료일을 수정하고 cfg를 다시 생성한 뒤 실행하세요. "
            "날짜를 자동 변경하거나 부족한 가격을 채우지는 않았습니다."
        )
    # Check dates for every formation length before launching expensive tests.
    for months in FORMATION_MONTHS:
        cfg = base.with_updates({"schedule.formation_months": months})
        make_windows(data.prices.index, cfg)
    data_hash = _data_fingerprint(data)
    identity = {"config": base.values, "data_sha256": data_hash, "source_sha256": source_fingerprint(),
        "merger_calendar_sha256": mergers.fingerprint, "merger_calendar_coverage": mergers.coverage,
        "plan": plan.to_dict("records"), "output_profile": output_profile,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "statsmodels")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if output_dir is None:
        root = Path(base["run"]["output_directory"]).expanduser()
        if not root.is_absolute():
            root = base.project_root / root
        output = root / "selection_study_144_mergers"
    else:
        output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigError("같은 결과 폴더에서 이미 실험이 실행 중입니다") from exc
        manifest_path = output / "study_manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ConfigError("이미 있는 실험 폴더입니다. resume=True 또는 새 output_dir를 지정하세요")
            state = json.loads(manifest_path.read_text(encoding="utf-8"))
            if state["fingerprint"] != fingerprint:
                raise ConfigError("기존 실험과 설정·데이터·코드가 다릅니다. 새 결과 폴더를 지정하세요")
        else:
            if any(p.name != ".study.lock" for p in output.iterdir()):
                raise ConfigError("기존 파일이 있는 폴더를 실험 결과로 덮어쓸 수 없습니다")
            state = {"fingerprint": fingerprint, "identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_mode": data.metadata.get("mode", "in_memory"),
                "evaluation_label": "exploratory_not_holdout" if base["evaluation"]["active_split"] == "all" else base["evaluation"]["active_split"],
                "experiments": [{**r, "status": "pending", "attempts": 0, "output_directory": "", "error": ""}
                    for r in plan.to_dict("records")]}
        for record in state["experiments"]:
            if record["status"] == "running":
                record["status"] = "interrupted"
            if record["status"] == "complete" and not _verify_completed(output, record):
                record.update(status="interrupted", error="완료 파일 검증 실패")
                record.pop("summary", None)
        _checkpoint(output, state)
        cache = SelectionCache(output / "selection_cache", data_hash)
        count, current_model = 0, None
        for record, cfg in zip(state["experiments"], configs):
            if record["model"] not in chosen or record["status"] == "complete":
                continue
            if record["status"] == "failed" and not retry_failed:
                continue
            if max_experiments is not None and count >= max_experiments:
                break
            if progress:
                if current_model != record["model"]:
                    current_model = record["model"]
                    progress(f"모델 {current_model}: {MODELS[current_model][2]} (모델당 {len(plan)//len(MODELS)}개)")
                progress(f"[{record['experiment']}/{len(plan)}] {record['name']}")
            record["attempts"] += 1
            trial = Path(f"model_{record['model']}") / record["name"] / f"attempt_{record['attempts']:03d}"
            record.update(status="running", output_directory=str(trial), error="")
            record.pop("summary", None)
            _checkpoint(output, state)
            try:
                result = run_backtest(cfg, data=data, output_dir=output / trial, progress=progress,
                    formation_cache=cache, output_profile=output_profile,
                    selection_audit_dir=output / "selection_audits" / f"{record['model']}_F{record['formation_months']}")
                record.update(status="complete", summary=result["summary"])
                del result
            except KeyboardInterrupt:
                record.update(status="interrupted", error="사용자 중단")
                _checkpoint(output, state)
                raise
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if progress:
                    progress(f"  실패 기록: {record['error']}")
            count += 1
            _checkpoint(output, state)
        table = _checkpoint(output, state)
        if progress:
            progress(f"저장: {output} / 완료 {state['counts']['complete']}/{len(plan)}, 실패 {state['counts']['failed']}")
        return table, output

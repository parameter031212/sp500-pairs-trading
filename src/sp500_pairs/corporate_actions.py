"""Point-in-time merger exclusions; announcement knowledge is separate from trading availability."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MergerEvent:
    event_id: str
    ticker: str
    announcement_date: pd.Timestamp
    timing: str
    last_trading_date: pd.Timestamp | None
    release_date: pd.Timestamp | None
    source: str
    trading_source: str
    release_source: str = ""

    @property
    def known_date(self):
        # Without a verified intraday timestamp, never use the news at that day's close.
        return self.announcement_date + pd.Timedelta(days=0 if self.timing == "before_close" else 1)


class MergerCalendar:
    def __init__(self, events=(), coverage="user_supplied"):
        self.events = tuple(sorted(events, key=lambda e: (e.known_date, e.event_id)))
        self.coverage = coverage
        rows = self.records()
        self.fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def records(self):
        return [{"event_id": e.event_id, "ticker": e.ticker,
            "announcement_date": str(e.announcement_date.date()), "timing": e.timing,
            "known_date": str(e.known_date.date()),
            "last_trading_date": str(e.last_trading_date.date()) if e.last_trading_date is not None else "",
            "release_date": str(e.release_date.date()) if e.release_date is not None else "",
            "source": e.source, "trading_source": e.trading_source,
            "release_source": e.release_source} for e in self.events]

    def for_tickers(self, *tickers):
        return MergerCalendar((e for e in self.events if e.ticker in tickers), self.coverage)

    def active(self, date):
        date = pd.Timestamp(date).normalize()
        return tuple(e for e in self.events if e.known_date <= date and (e.release_date is None or date < e.release_date))

    def tradable(self, date):
        # Used only at the current execution date, never to schedule an earlier exit.
        return not any(e.last_trading_date is not None and pd.Timestamp(date) > e.last_trading_date for e in self.events)


def read_merger_calendar(path):
    from .data import DataError, normalize_ticker
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"event_id", "ticker", "announcement_date", "timing", "last_trading_date", "release_date", "source", "trading_source"}
    if not required <= set(frame) or frame.event_id.duplicated().any():
        raise DataError("합병 일정의 필수 열 누락 또는 event_id 중복")
    events = []
    for row in frame.to_dict("records"):
        if row["timing"] not in {"before_close", "after_close", "date_only"}:
            raise DataError("합병 발표 timing: before_close/after_close/date_only")
        if not row["event_id"] or not row["ticker"] or not row["source"].startswith("https://"):
            raise DataError("합병 일정에는 사건 ID·종목·공식 출처가 필요합니다")
        def date(key):
            if row[key] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[key]):
                raise DataError(f"합병 일정의 {key}는 YYYY-MM-DD 형식")
            try:
                value = pd.Timestamp(row[key]) if row[key] else None
            except ValueError as exc:
                raise DataError(f"합병 일정의 {key} 확인: {row[key]}") from exc
            if value is not None and (pd.isna(value) or value.tzinfo is not None):
                raise DataError(f"합병 일정의 {key}는 YYYY-MM-DD 형식")
            return value
        announced, last, release = date("announcement_date"), date("last_trading_date"), date("release_date")
        if announced is None or (last is not None and last < announced):
            raise DataError("합병 발표일/마지막 거래일 확인")
        if release is not None and (release <= announced or last is not None):
            raise DataError("해제일은 발표일 이후여야 하며 거래종료 사건에는 사용하지 않습니다")
        if release is not None and not row.get("release_source", "").startswith("https://"):
            raise DataError("합병 취소에 따른 해제일에는 공식 출처가 필요합니다")
        if last is not None and not row["trading_source"].startswith("https://"):
            raise DataError("마지막 거래일에는 별도 공식 출처가 필요합니다")
        events.append(MergerEvent(row["event_id"], normalize_ticker(row["ticker"]), announced,
            row["timing"], last, release, row["source"], row["trading_source"], row.get("release_source", "")))
    return MergerCalendar(events, "curated_announcements_not_exhaustive")


@lru_cache(maxsize=8)
def _read_cached(path, modified_ns, size):
    return read_merger_calendar(path)


def configured_calendar(cfg):
    from .data import DataError
    if not cfg["corporate_actions"]["enabled"]:
        return MergerCalendar()
    path = Path(cfg["corporate_actions"]["events_file"]).expanduser()
    if not path.is_absolute():
        path = cfg.project_root / path
    if not path.is_file():
        raise DataError(f"합병 일정 파일이 없습니다: {path}")
    stat = path.stat()
    return _read_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def prepare_corporate_actions(data, cfg):
    if cfg["corporate_actions"]["enabled"]:
        # Explicit in-memory calendars are supported for reproducible tests/research.
        if data.mergers is None or data.metadata.get("merger_calendar_from_config"):
            data.mergers = configured_calendar(cfg)
            data.metadata["merger_calendar_from_config"] = True
        data.metadata["merger_policy"] = {
            "sha256": data.mergers.fingerprint, "events": len(data.mergers.events),
            "coverage": data.mergers.coverage,
            "rule": "block targets once announcement known; exit both legs at next tradable close",
            "unknown_time": "recognize from the first session after the announcement date",
        }
    else:
        data.metadata.pop("merger_policy", None)
    return data


def calendar_for(data, cfg):
    prepare_corporate_actions(data, cfg)
    return data.mergers if cfg["corporate_actions"]["enabled"] else MergerCalendar()

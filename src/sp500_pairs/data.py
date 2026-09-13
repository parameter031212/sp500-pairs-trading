from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import os

import numpy as np
import pandas as pd

from .config import Config


class DataError(ValueError):
    pass


def normalize_ticker(value: str) -> str:
    value = str(value).strip().upper()
    if value.endswith(".US"):
        value = value[:-3]
    return value.replace(".", "-").replace("/", "-")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class MarketData:
    prices: pd.DataFrame
    membership: pd.DataFrame
    sectors: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.prices = self.prices.copy()
        self.prices.index = pd.DatetimeIndex(self.prices.index).normalize()
        self.prices.columns = [normalize_ticker(x) for x in self.prices.columns]
        if self.prices.index.has_duplicates or self.prices.columns.has_duplicates:
            raise DataError("가격 날짜 또는 정규화된 티커 중복")
        self.prices = self.prices.sort_index().astype(float)
        if self.prices.empty or np.isinf(self.prices.to_numpy()).any() or (self.prices <= 0).any().any():
            raise DataError("빈 가격 데이터 또는 0 이하/무한 가격. 원본 확인이 필요합니다.")
        self.membership = self.membership.copy()
        self.membership["date"] = pd.to_datetime(self.membership["date"]).dt.normalize()
        if self.membership.empty or self.membership["date"].duplicated().any():
            raise DataError("구성종목 스냅샷이 없거나 같은 날짜가 중복됩니다")
        self.membership = self.membership.sort_values("date").reset_index(drop=True)
        self.membership["members"] = self.membership["tickers"].map(self._members)
        self.sectors = self.sectors.copy()
        self.sectors["ticker"] = self.sectors["ticker"].map(normalize_ticker)
        self.sectors["sector"] = self.sectors["sector"].fillna("").astype(str).str.strip()
        if "date" in self.sectors:
            self.sectors["date"] = pd.to_datetime(self.sectors["date"]).dt.normalize()
            self.sectors = self.sectors.sort_values(["date", "ticker"])
            keys = ["date", "ticker"]
        else:
            keys = ["ticker"]
        if self.sectors.duplicated(keys).any():
            raise DataError("섹터 날짜/티커가 중복됩니다. 임의로 한 행을 선택하지 않습니다.")

    @staticmethod
    def _members(value):
        if not isinstance(value, str) or not value.strip():
            raise DataError("구성종목은 비어 있지 않은 쉼표 구분 문자열이어야 합니다")
        return tuple(sorted({normalize_ticker(x) for x in value.split(",") if x.strip()}))

    def universe(self, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
        history = self.membership.loc[self.membership["date"] <= as_of]
        if history.empty:
            raise DataError(f"{as_of.date()} 이전의 S&P500 구성종목 자료가 없습니다")
        snapshot = history.iloc[-1]
        sectors = self.sectors
        if "date" in sectors:
            sectors = sectors.loc[sectors["date"] <= as_of].drop_duplicates("ticker", keep="last")
        mapping = sectors.set_index("ticker")["sector"]
        rows = [{"ticker": t, "sector": mapping.get(t, "")} for t in snapshot["members"]]
        return pd.DataFrame(rows), {
            "membership_snapshot": str(snapshot["date"].date()),
            "membership_count": len(rows),
            "sector_mode": "dated_asof" if "date" in self.sectors else "static_not_historical",
        }


def load_data(cfg: Config, data_dir: str | Path | None = None) -> MarketData:
    if cfg["data"]["mode"] == "synthetic":
        return synthetic_data(cfg["run"]["seed"])
    d = cfg["data"]
    directory = Path(data_dir or os.environ.get("SP500_DATA_DIR") or d["directory"]).expanduser()
    if not directory.is_absolute():
        directory = cfg.project_root / directory
    paths = {key: directory / d[key + "_file"] for key in ("prices", "membership", "sectors")}
    for path in paths.values():
        if not path.is_file():
            raise DataError(f"입력 파일이 없습니다: {path}. --data-dir 또는 SP500_DATA_DIR를 지정하세요.")
    prices = pd.read_csv(paths["prices"])
    if "date" not in prices:
        raise DataError("가격 CSV는 date와 종목별 가격 열이 필요합니다")
    prices = prices.set_index(pd.to_datetime(prices.pop("date"), errors="raise"))
    membership = pd.read_csv(paths["membership"])
    sectors = pd.read_csv(paths["sectors"])
    if not {"date", "tickers"} <= set(membership):
        raise DataError("구성종목 CSV는 date,tickers 열이 필요합니다")
    if not {"ticker", "sector"} <= set(sectors):
        raise DataError("섹터 CSV는 ticker,sector 열이 필요합니다 (과거 자료는 date 추가)")
    metadata = {
        "mode": "csv",
        "inputs": {key: {"name": p.name, "sha256": sha256(p)} for key, p in paths.items()},
        "price_basis": "supplied_adjusted_close_proxy; corporate-action/dividend coverage requires vendor validation",
        "membership_interpretation": "snapshot effective on date; publication lag not present in input",
        "sector_mode": "dated_asof" if "date" in sectors else "static_not_historical",
        "warnings": [
            "Adjusted prices are research execution proxies, not a reconstruction of raw shares and corporate actions.",
            "Ticker normalization is not a permanent security identifier; mergers and ticker reuse require source validation.",
            "Missing held prices stop the run; no fictional liquidation at a stale price is performed.",
        ],
    }
    if "date" not in sectors:
        metadata["warnings"].append("Static sector mapping does not establish historical GICS membership or an 11-sector taxonomy.")
    result = MarketData(prices, membership, sectors, metadata)
    metadata.update({
        "price_start": str(result.prices.index.min().date()),
        "price_end": str(result.prices.index.max().date()),
        "price_rows": len(result.prices), "price_columns": len(result.prices.columns),
        "sector_labels": sorted(result.sectors["sector"].unique().tolist()),
        "large_one_day_move_count_over_50pct": int((result.prices.pct_change(fill_method=None).abs() > .5).sum().sum()),
    })
    return result


def synthetic_data(seed: int = 20260911) -> MarketData:
    """Known illustrative processes for software checks; never reported as S&P500 results."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", "2021-01-15")
    values, sectors = {}, []
    for sector in range(6):
        common = np.log(50 + sector * 10) + np.cumsum(rng.normal(0.0001, .012, len(dates)))
        for stock in range(4):
            residual = np.zeros(len(dates))
            shocks = rng.normal(0, .005, len(dates))
            for i in range(1, len(dates)):
                residual[i] = .93 * residual[i-1] + shocks[i]
            if stock == 3:
                residual[dates >= "2017-04-03"] += .18
            ticker = f"S{sector}{stock}"
            values[ticker] = np.exp(common + .15 * stock + residual)
            sectors.append({"ticker": ticker, "sector": f"Synthetic sector {sector}"})
    return MarketData(
        pd.DataFrame(values, index=dates),
        pd.DataFrame([{"date": "2014-01-01", "tickers": ",".join(values)}]),
        pd.DataFrame(sectors),
        {"mode": "synthetic", "seed": seed, "warnings": ["Business-day calendar and invented prices; this is not a market backtest."]},
    )

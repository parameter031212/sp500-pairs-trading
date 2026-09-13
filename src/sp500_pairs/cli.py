import argparse
import json
import tomllib

from .config import load_config
from .data import load_data
from .runner import json_safe, run_backtest, run_grid


def main(argv=None):
    parser = argparse.ArgumentParser(description="팀 공통 S&P500 페어트레이딩 베이스라인")
    parser.add_argument("command", choices=["run", "grid", "check-data"], nargs="?", default="run")
    parser.add_argument("--config", default="configs/baseline.toml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args(argv)
    if args.command == "grid":
        if args.set or args.max_windows is not None:
            parser.error("grid의 기본 설정과 제한은 grid가 참조하는 설정 파일에서 지정하세요")
        _, destination = run_grid(args.config, args.data_dir)
        print(f"실험 비교표: {destination / 'comparison.csv'}")
        return
    cfg = load_config(args.config)
    updates = {}
    for value in args.set:
        key, separator, raw = value.partition("=")
        if not separator:
            parser.error("--set은 section.key=value 형식입니다")
        updates[key] = tomllib.loads("value = " + raw)["value"]
    if args.max_windows is not None:
        updates["run.max_windows"] = args.max_windows
    cfg = cfg.with_updates(updates)
    if args.command == "check-data":
        data = load_data(cfg, args.data_dir)
        print(json.dumps(json_safe(data.metadata), ensure_ascii=False, indent=2))
        return
    result = run_backtest(cfg, data_dir=args.data_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"결과 폴더: {result['output_directory']}")

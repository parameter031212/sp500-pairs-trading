# S&P500 팀 공통 페어트레이딩 베이스라인

같은 매매·손익 계산 코드를 사용하면서, 팀원별로 설정 파일을 바꿔 실험하는 연구용 모델입니다.
S&P500 과거 구성종목에서 같은 섹터의 페어를 찾고, 로그가격 Engle–Granger 검정과 고정된 스프레드로 거래합니다.

**먼저 볼 파일**

| 목적 | 파일 |
|---|---|
| 노트북에서 실행 | [notebooks/Start_Here.ipynb](notebooks/Start_Here.ipynb) |
| 팀 공통 기본값 | [configs/baseline.toml](configs/baseline.toml) |
| B 모델 별도 실행 설정 | [configs/model_b.toml](configs/model_b.toml) |
| B 모델 선정·거래 규칙 | [docs/model_b.md](docs/model_b.md) |
| 개인 실험 설정 예시 | [configs/example_experiment.toml](configs/example_experiment.toml) |
| 양방향 비교 옵션 | [configs/bidirectional.toml](configs/bidirectional.toml) |
| 여러 조합을 한 번에 비교 | [configs/grid_demo.toml](configs/grid_demo.toml) |
| 정확한 거래·회계 규칙 | [docs/specification.md](docs/specification.md) |
| 합의사항과 미정 항목 | [docs/decisions.md](docs/decisions.md) |
| 데이터 형식·한계 | [data/README.md](data/README.md) |
| B 모델 합병 발표 규칙 | [reference_data/README.md](reference_data/README.md) |
| 탐지 전략 연결 방법 | [docs/detector_extension.md](docs/detector_extension.md) |
| 배포 전 검증 범위 | [docs/validation.md](docs/validation.md) |

## 빠른 시작

기존 공통 모델은 `configs/baseline.toml`에 유지합니다. B 모델은
`configs/model_b.toml`을 지정해 별도로 실행합니다. B는 공적분·일별 로그수익률
상관계수와 기존 거래 규칙을 사용하며, 종목·섹터별 페어 수 제한을 해제하고
확인된 합병 발표에 대한 거래 규칙을 적용합니다. 다중검정 보정과 정성적 기업
검수는 적용하지 않습니다. B의 정확한 조건과 자료 한계는
[B 모델 안내](docs/model_b.md)에 기록했습니다.

Python 3.11 이상을 사용합니다. 이 저장소 폴더를 열고 다음을 실행하세요.

```bash
python -m pip install -r requirements.txt
python run_backtest.py run --config configs/demo.toml
```

이 명령은 **가상의 가격으로 전체 실행 과정을 확인**합니다. S&P500 투자 성과가 아닙니다.
결과는 `results/demo/실행시각_식별자/`에 저장됩니다.

노트북을 사용하려면 Jupyter 환경에서 `notebooks/Start_Here.ipynb`를 엽니다.
필요하면 `python -m pip install -e ".[notebook]"`으로 노트북 환경까지 설치할 수 있습니다.
노트북도 동일한 공통 코드를 호출하며, 별도의 매매 구현을 포함하지 않습니다.

## 실제 데이터로 실행

다음 세 파일을 `data/`에 두거나 데이터 폴더를 지정합니다.

```text
SP500_adjusted_close_cleaned_2005-09-27_2025-12-31.csv
S&P 500 Historical Components & Changes (Updated).csv
SP500_sector_mapping.csv
```

```bash
python run_backtest.py check-data --config configs/baseline.toml --data-dir "/path/to/data"
python run_backtest.py run --config configs/baseline.toml --data-dir "/path/to/data" --max-windows 1
```

전체 18개 구간을 실행하려면 `--max-windows 1`을 빼세요. 한 구간의 모든 후보 페어를 검정하므로 실제 데이터는 데모보다 오래 걸립니다.
파일명이 다르면 `configs/local.toml`에 `extends = "baseline.toml"`과 `[data]`의 파일명을 적으면 됩니다.
원본 가격·구성종목·섹터 파일은 수정하지 않습니다. 데이터 파일, 개인 경로 설정, 실행 결과는 Git에서 제외됩니다.

## 파라미터 바꾸기

공통 기본 설정은 보존하고 `configs/example_experiment.toml`을 개인 설정 파일로 복사하세요.

```toml
extends = "baseline.toml"

[schedule]
formation_months = 24
trading_months = 6

[strategy]
entry_z = 2.5
exit_z = 0.5
stop_enabled = true
stop_z = 4.0
max_holding_days = 90

[run]
output_directory = "results/my_experiment"
```

```bash
python run_backtest.py run --config configs/example_experiment.toml --data-dir "/path/to/data"
```

**손절은 3.5로 고정되지 않습니다.** `stop_z`를 바꾸거나 `stop_enabled = false`로 끌 수 있습니다.
손절 후 재진입까지 실험하려면 `reentry_after_stop = "after_reset"`으로 바꿉니다.
이 옵션은 청산 이후 `|Z| < entry_z`로 돌아온 것을 먼저 관측하고, 그 다음 새 진입 조건을 기다립니다.

| 조정할 항목 | 설정 |
|---|---|
| 추정·거래기간 | `schedule.formation_months`, `schedule.trading_months` |
| 진입·청산 기준 | `strategy.entry_z`, `strategy.exit_z` |
| 손절 사용·기준 | `strategy.stop_enabled`, `strategy.stop_z` |
| 보유기간 제한 | `strategy.max_holding_enabled`, `strategy.max_holding_days` |
| 손절·시간청산 후 재진입 | `strategy.reentry_after_stop`, `strategy.reentry_after_time_exit` |
| 공적분 유의수준·회귀 방향 | `formation.cointegration_alpha`, `formation.orientation` |
| 상관·헤지비율·반감기 | `formation`의 관련 항목 |
| 최대 페어 수·집중 제한 | `selection`의 세 항목 |
| 거래·슬리피지·대차비용 | `costs`의 세 항목 |

청산 기준의 부호에 주의하세요. `exit_z = 0.5`는 평균에 도달하기 전 청산, `0`은 평균 통과, `-0.5`는 평균을 지나 반대편까지 기다리는 설정입니다.
정확히 0을 찍을 필요는 없습니다. 잘못된 키나 손절 기준이 진입 기준보다 작은 조합은 실행 전에 오류로 알립니다.

## 파라미터 조합 비교

```bash
python run_backtest.py grid --config configs/grid_demo.toml
```

예시는 가상 자료에서 진입 2개 값과 손절 2개 값을 조합한 4회 실험입니다.
`comparison.csv`와 각 실험의 전체 설정·거래 기록을 저장합니다. 공통 포메이션 계산은 실행 중 재사용합니다.
실제 데이터에서는 `base`를 해당 설정 파일로 바꾸세요.

학습·검증·시험 날짜는 아직 팀에서 확정하지 않았습니다. 기본 `active_split = "all"` 결과는 **exploratory_not_holdout**으로 표시합니다.
`evaluation`에 구간을 채우고 `active_split = "validation"`처럼 지정할 수 있습니다. 날짜는 거래 구간 경계에 맞추세요.
최종 `test` 구간에서는 grid 탐색을 허용하지 않습니다. 일반 단일 실행은 가능합니다.
같은 파라미터에서 탐지 없음/A/B를 비교해야 탐지의 추가 효과를 해석할 수 있습니다.

## 결과 읽기

| 결과 | 내용 |
|---|---|
| `effective_config.json` | 실제 적용된 전체 설정 |
| `manifest.json` | 데이터 파일 해시, 코드 해시, 라이브러리 버전, 실행 범위·상태·한계 |
| `summary.json` | 전체 달력과 배정자본 기준의 성과 |
| `portfolio_daily.csv` | 빈 슬롯까지 포함한 계좌 NAV·비용·수익률 |
| `pair_daily.csv` | 보유 수량, 신호, 체결, 청산 후 모니터링 |
| `trades.csv` | 신호일과 체결일을 구분한 거래별 손익 |
| `tested_pairs.csv` | 통과·탈락·집중 제한 이유를 포함한 후보군 |
| `direction_tests.csv` | 방향별 EG 결과와 추정계수 |
| `universe_audit.csv` | 당시 구성종목, 포메이션 커버리지·제외 이유·I(1) 진단 |
| `windows.csv` | 각 추정·거래 구간, 선정 수, 슬롯자본 |
| `stock_exposure.csv` | 종목별 순노출과 페어 계좌별 총노출 합계 |
| `equity_and_drawdown.png` | 자산곡선·최대낙폭 경로 |

실행이 중단되면 `manifest.json`은 `failed`이고 이유가 기록됩니다. 실패한 실행을 완료된 성과로 사용하지 않습니다.

## 협업

1. 공통 기본 설정과 거래·회계 코드를 기준 버전으로 유지합니다.
2. 개인 실험은 별도 설정 파일과 실험 목적을 함께 공유합니다.
3. 매매 엔진을 수정했다면 아래 검증을 실행하고 변경 이유를 기록합니다.
4. GitHub에는 이 폴더의 내용을 저장소 루트로 올리면 됩니다. 원본 데이터와 개인 실행 결과는 제외됩니다.

```bash
python -m unittest discover -s tests -v
```

GitHub Actions에도 같은 검증이 설정되어 있습니다. 이것은 소프트웨어 검증이며 실증 수익성을 입증하지 않습니다.

## 연구상 범위

Engle–Granger 검정과 후속 문헌의 거래 구조를 결합한 팀의 기준모형입니다. Clegg–Krauss의 CI1을 그대로 복제한 모형은 아닙니다.
고정 섹터 매핑, 공급 가격의 조정 방식, 상장폐지·티커 재사용, 대차비용 가정은 입력 데이터의 검증과 함께 해석해야 합니다.
가격 결측이 있는 보유 포지션은 오래된 가격으로 임의 청산하지 않고 실행을 중단합니다.

참고: [Engle–Granger 구현 문서](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html),
[다중검정 보정](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html),
[Clegg–Krauss (2018)](https://doi.org/10.1080/14697688.2017.1370122).

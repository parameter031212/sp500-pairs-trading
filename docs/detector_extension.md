# 공통 엔진에 탐지 전략 연결하기

기본 모델에는 실제 공적분 붕괴 탐지나 PCI가 들어 있지 않습니다. 탐지 전략의 후속 비교를 위한 연결 지점만 제공합니다.
`detector_factory(pair, formation_prices)`는 선정된 페어마다 새로운 탐지기 객체를 만듭니다.
전달되는 가격은 포메이션 구간뿐입니다. 이후에는 `on_close(observation)`을 매 거래일 순차 호출합니다.

```python
from sp500_pairs import load_config, run_backtest

class MyDetector:
    def __init__(self, pair, formation_prices):
        self.history = []

    def on_close(self, observation):
        self.history.append((observation.date, observation.zscore))
        # 여기에서 그날까지의 자료로 팀원의 탐지 통계량을 계산합니다.
        return False

cfg = load_config("configs/baseline.toml")
baseline = run_backtest(cfg)
with_detector = run_backtest(cfg, detector_factory=MyDetector)
```

위 예시는 항상 False를 반환하는 연결 예시이며 탐지 전략 자체가 아닙니다.
True는 보유 중이면 다음 종가 청산을 요청하고, 미보유 상태를 포함해 해당 구간의 새 진입을 막습니다.
같은 날의 기존 예정 주문 체결보다 탐지 결과를 먼저 알아낸 것으로 취급하지 않습니다.
청산 후에도 모니터링은 원래 거래 구간 종료까지 계속됩니다. 포지션 종료를 공적분 상태의 정답으로 취급하지 않습니다.

탐지 비교에서는 같은 데이터·기본 파라미터·손절·시간 제한·비용을 사용하세요.
이 API는 동일한 청산·거래중단 대응의 비교를 위한 것입니다. 계수 재추정이나 PCI 전환은 별도의 대응 모듈 설계가 필요합니다.
사용자 정의 탐지기의 코드는 별도로 버전 관리해야 합니다. 기본 manifest의 코드 해시는 공통 패키지만 포함하며 외부 탐지기 코드를 자동 수집하지 않습니다.

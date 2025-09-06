from datetime import datetime, timedelta
import sys, types, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from models import MarketData, MarketDataResponse, AlphaSpec, AlphaOp


class DummyCollector:
    def get_market_data(self, symbols, timeframe, period):
        start = datetime(2024, 1, 1)
        data = [
            MarketData(symbol="AAA", timestamp=start + timedelta(days=i), open=10 + i, high=10 + i, low=10 + i, close=10 + i, volume=1000)
            for i in range(3)
        ]
        return MarketDataResponse(
            symbols_analyzed=symbols,
            market_data=data,
            market_trend="neutral",
            volatility_assessment="low",
            key_insights=[],
            timestamp=datetime.now(),
        )


sys.modules['market_data_tools'] = types.ModuleType('market_data_tools')
sys.modules['market_data_tools'].MarketDataCollector = DummyCollector

import alpha_engine


def test_compute_alphas_smoke():
    spec = AlphaSpec(
        name="delta1",
        input="close",
        ops=[AlphaOp(name="delta", k=1, window=1)],
    )
    result = alpha_engine.compute_alphas(["AAA"], [spec], "1d", "3d")
    assert result["success"]
    assert result["reports"][0]["factor"] == "delta1"

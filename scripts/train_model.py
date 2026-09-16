from __future__ import annotations

from pathlib import Path

from quant_system.backtest.engine import backtest
from quant_system.config import load_config
from quant_system.data.loader import load_symbol_data
from quant_system.features.core import build_feature_frame
from quant_system.models.core import probability_to_signal, train_logistic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    data_cfg, feature_cfg, model_cfg = config["data"], config["features"], config["model"]
    frame = load_symbol_data(data_cfg["symbols"][0], ROOT / "data" / "raw")
    X, y, columns = build_feature_frame(
        frame,
        feature_cfg["return_windows"],
        feature_cfg["volatility_windows"],
        feature_cfg["moving_average_fast"],
        feature_cfg["moving_average_slow"],
        model_cfg["horizon"],
    )
    model, probability, _, model_metrics = train_logistic(
        X, y, model_cfg["test_size"], model_cfg["random_state"]
    )
    result = backtest(
        frame.loc[probability.index],
        probability_to_signal(probability, model_cfg["threshold"]),
        config["backtest"]["transaction_cost_bps"],
        config["backtest"]["initial_capital"],
    )
    print("ML:", model_metrics)
    print("Backtest:", result.metrics)
    print("Features:", columns)


if __name__ == "__main__":
    main()

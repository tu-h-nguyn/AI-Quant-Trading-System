from __future__ import annotations

from pathlib import Path

from quant_system.config import load_config
from quant_system.data.downloader import download_ohlcv, save_symbol_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    data_cfg = config["data"]
    frames = download_ohlcv(
        data_cfg["symbols"],
        data_cfg["start"],
        data_cfg.get("end"),
        data_cfg["interval"],
    )
    for symbol, frame in frames.items():
        print(save_symbol_data(frame, symbol, ROOT / "data" / "raw"))


if __name__ == "__main__":
    main()

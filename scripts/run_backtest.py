from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from quant_system.config import load_config
from quant_system.data.loader import load_symbol_data
from quant_system.strategies.momentum import momentum
from quant_system.strategies.moving_average import moving_average
from quant_system.backtest.engine import backtest

def main():
 c=load_config("configs/default.yaml"); symbol=c["data"]["symbols"][0]; df=load_symbol_data(symbol)
 strategies={"Momentum":momentum(df,20),"MovingAverage":moving_average(df,c["features"]["moving_average_fast"],c["features"]["moving_average_slow"])}
 for name,s in strategies.items():
  r=backtest(df,s,c["backtest"]["transaction_cost_bps"],c["backtest"]["initial_capital"]); print(name,r.metrics)
if __name__=="__main__": main()

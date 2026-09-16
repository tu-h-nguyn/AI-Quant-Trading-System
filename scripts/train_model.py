from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from quant_system.config import load_config
from quant_system.data.loader import load_symbol_data
from quant_system.models.features import build_feature_frame
from quant_system.models.train import train_logistic
from quant_system.models.predict import probability_to_signal
from quant_system.backtest.engine import backtest

def main():
 c=load_config("configs/default.yaml"); d=c["data"]; f=c["features"]; m=c["model"]; df=load_symbol_data(d["symbols"][0])
 X,y,cols=build_feature_frame(df,f["return_windows"],f["volatility_windows"],f["moving_average_fast"],f["moving_average_slow"],m["horizon"])
 model,pred,labels,metrics=train_logistic(X,y,m["test_size"],m["random_state"]); print("ML:",metrics)
 sig=probability_to_signal(pred,m["threshold"]); r=backtest(df.loc[pred.index],sig,c["backtest"]["transaction_cost_bps"],c["backtest"]["initial_capital"]); print("Backtest:",r.metrics); print("Features:",cols)
if __name__=="__main__": main()

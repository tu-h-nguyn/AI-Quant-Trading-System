from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from quant_system.config import load_config
from quant_system.data.downloader import download_ohlcv,save_symbol_data

def main():
 c=load_config("configs/default.yaml"); d=c["data"]; data=download_ohlcv(d["symbols"],d["start"],d.get("end"),d["interval"])
 for s,df in data.items(): print(save_symbol_data(df,s))
if __name__=="__main__": main()

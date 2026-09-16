import pandas as pd

def inverse_volatility_weights(volatility):
    inv=1/volatility.replace(0,pd.NA)
    return inv.div(inv.sum(axis=1),axis=0).fillna(0)

def cap_weights(weights,max_weight=.2):
    w=weights.clip(lower=0,upper=max_weight); total=w.sum(); return w/total if total>0 else w

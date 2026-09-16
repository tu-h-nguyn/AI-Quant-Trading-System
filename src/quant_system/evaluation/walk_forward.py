import pandas as pd

def walk_forward_predict(model, X, y, test_window=63, min_train_size=252):
    if len(X)!=len(y): raise ValueError("X and y must have same length")
    predictions=[]
    for start in range(min_train_size,len(X),test_window):
        end=min(start+test_window,len(X)); model.fit(X.iloc[:start],y.iloc[:start])
        p=model.predict_proba(X.iloc[start:end])[:,1]
        predictions.append(pd.Series(p,index=X.index[start:end]))
    return pd.concat(predictions).sort_index() if predictions else pd.Series(dtype=float)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

def chronological_split(X,y,test_size=.2):
    split=int(len(X)*(1-test_size)); return X.iloc[:split],X.iloc[split:],y.iloc[:split],y.iloc[split:]

def train_logistic(X,y,test_size=.2,random_state=42):
    Xtr,Xte,ytr,yte=chronological_split(X,y,test_size)
    model=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1000,random_state=random_state))])
    model.fit(Xtr,ytr); p=pd.Series(model.predict_proba(Xte)[:,1],index=Xte.index,name="probability")
    pred=(p>=.5).astype(int)
    return model,p,pred,{"accuracy":float(accuracy_score(yte,pred)),"roc_auc":float(roc_auc_score(yte,p))}

def probability_to_signal(p,threshold=.5): return (p>=threshold).astype(float).rename("signal")

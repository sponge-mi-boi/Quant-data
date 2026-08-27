import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import torch
from torch import nn


class RegimeTransformer(nn.Module):
    def __init__(self,n_features,n_classes,d_model,layers,dropout):
        super().__init__(); self.input=nn.Linear(n_features,d_model); self.position=nn.Parameter(torch.zeros(1,20,d_model))
        block=nn.TransformerEncoderLayer(d_model=d_model,nhead=2,dim_feedforward=2*d_model,dropout=dropout,batch_first=True,activation='gelu',norm_first=True)
        self.encoder=nn.TransformerEncoder(block,num_layers=layers); self.output=nn.Linear(d_model,n_classes)
    def forward(self,x):
        z=self.input(x)+self.position[:,:x.shape[1]]; return self.output(self.encoder(z)[:,-1])


def _sequences(frame,lookback=20):
    values=frame.to_numpy(dtype=np.float32); rows=[]; positions=[]
    for i in range(lookback-1,len(frame)):
        if np.isfinite(values[i-lookback+1:i+1]).all(): rows.append(values[i-lookback+1:i+1]); positions.append(i)
    return np.asarray(rows,dtype=np.float32),positions


def fit_transformer(features,labels,d_model=8,layers=1,dropout=.1,seed=41):
    torch.manual_seed(seed); x,y=pd.DataFrame(features).align(pd.DataFrame(labels),join='inner',axis=0); mean=x.mean(); scale=x.std().replace(0,1.); z=(x-mean)/scale
    seq,pos=_sequences(z); target=y.iloc[pos].idxmax(axis=1); states=list(labels.columns); encoded=target.map({n:i for i,n in enumerate(states)}).to_numpy(dtype=np.int64)
    split=max(1,int(.85*len(seq))); tx=torch.tensor(seq[:split]); ty=torch.tensor(encoded[:split]); vx=torch.tensor(seq[split:]); vy=torch.tensor(encoded[split:])
    model=RegimeTransformer(x.shape[1],len(states),d_model,layers,dropout); optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-3); loss=nn.CrossEntropyLoss(); best=None; best_loss=float('inf'); patience=0
    for _ in range(120):
        model.train(); optimizer.zero_grad(); loss(model(tx),ty).backward(); optimizer.step(); model.eval()
        with torch.no_grad(): score=float(loss(model(vx),vy)) if len(vx) else float(loss(model(tx),ty))
        if score<best_loss-1e-4: best_loss=score; best={k:v.detach().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        if patience>=15: break
    model.load_state_dict(best); model.eval(); return {'model':model,'mean':mean,'scale':scale,'columns':list(x.columns),'states':states}


def predict_transformer(fitted,features):
    x=pd.DataFrame(features).reindex(columns=fitted['columns']); seq,pos=_sequences((x-fitted['mean'])/fitted['scale']); index=x.index[pos]
    with torch.no_grad(): values=torch.softmax(fitted['model'](torch.tensor(seq)),dim=1).numpy()
    return pd.DataFrame(values,index=index,columns=fitted['states'])

# coding: utf-8
"""Explicit, user-selected fusion and diagnostic metrics on aligned branch predictions.
Requires labels for cohort AUC and a fixed threshold for classification statistics.
Research prototype; not an end-to-end clinical device.
"""
import argparse,json,sys
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss,confusion_matrix

def read(path):
 d=pd.read_csv(path,dtype={'ID':str})
 if 'ID' not in d or d.ID.duplicated().any() or d.ID.isna().any():raise ValueError('Invalid case IDs')
 return d.set_index('ID')
def ratio(a,b):return float(a/b) if b else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--tabular',required=True);p.add_argument('--habitat');p.add_argument('--image',required=True);p.add_argument('--method',choices=['mean5','median5','median4_no25','mean3_nohab','median3_clin_hab_2d'],required=True);p.add_argument('--output',required=True);p.add_argument('--labels');p.add_argument('--threshold',type=float);p.add_argument('--metrics');a=p.parse_args()
 t=read(a.tabular);i=read(a.image)
 if set(t.index)!=set(i.index):raise ValueError('Branch patient sets differ')
 d=t.join(i,how='left')
 if a.habitat:
  h=read(a.habitat)
  if set(t.index)!=set(h.index):raise ValueError('Branch patient sets differ')
  d=d.join(h,how='left')
 elif a.method!='mean3_nohab':raise ValueError('Habitat branch required for selected method')
 cols=['clinical_probability']
 if a.method!='median3_clin_hab_2d':cols.append('traditional_probability')
 if a.method!='mean3_nohab':cols.append('habitat_probability')
 cols.append('prob_2d')
 if a.method in ('mean5','median5'):cols.append('prob_25d')
 if not set(cols).issubset(d.columns):raise ValueError('Missing branch probabilities')
 v=d[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
 if not np.isfinite(v).all() or not ((v>=0)&(v<=1)).all():raise ValueError('Invalid branch probabilities')
 z=np.mean(v,axis=1) if a.method in ('mean5','mean3_nohab') else np.median(v,axis=1)
 d['fusion_probability']=z
 if a.threshold is not None:
  if not (0<=a.threshold<=1):raise ValueError('Threshold must be in [0,1]')
  d['classification']=(z>=a.threshold).astype(int)
 from pathlib import Path
 if Path(a.output).exists() or (a.metrics and Path(a.metrics).exists()):raise FileExistsError('Output exists; choose new paths')
 d.index.name='ID'
 report={'method':a.method,'n':len(d),'threshold':a.threshold,'onekey_loaded':any(x=='onekey_algo' or x.startswith('onekey_algo.') for x in sys.modules)}
 if a.labels:
  l=read(a.labels)
  if set(l.index)!=set(d.index) or 'y' not in l:raise ValueError('Labels do not match case IDs or lack y')
  raw_y=pd.to_numeric(l.loc[d.index,'y'],errors='raise').to_numpy(float)
  if not np.isfinite(raw_y).all() or not np.isin(raw_y,[0,1]).all():raise ValueError('Labels must be exactly 0 or 1')
  y=raw_y.astype(int)
  report.update(positive=int(y.sum()),negative=int(len(y)-y.sum()),auc=float(roc_auc_score(y,z)) if len(np.unique(y))==2 else None,brier=float(brier_score_loss(y,z)),log_loss=float(log_loss(y,z,labels=[0,1])))
  if a.threshold is not None:
   tn,fp,fn,tp=map(int,confusion_matrix(y,z>=a.threshold,labels=[0,1]).ravel());report.update(tp=tp,fp=fp,tn=tn,fn=fn,accuracy=ratio(tp+tn,len(y)),sensitivity=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),ppv=ratio(tp,tp+fp),npv=ratio(tn,tn+fn),f1=ratio(2*tp,2*tp+fp+fn))
 d.to_csv(a.output)
 if a.metrics:
  with open(a.metrics,'w',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()




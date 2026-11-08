# coding: utf-8
"""Standalone two-branch prototype: freeze training statistics, then infer new cases.
Requires scikit-learn-compatible models; no OneKey import. Not a complete OHP application.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd

def identity(v):
 s=str(v).strip()
 for suffix in ('.nii.gz','.nii.png','.nii.npy','.nii','.png','.npy'):
  if s.lower().endswith(suffix):return s[:-len(suffix)]
 return s

def read(path):
 d=pd.read_csv(path,dtype={'ID':str});d['ID']=d.ID.map(identity)
 if d.ID.isna().any() or d.ID.duplicated().any():raise ValueError('Invalid or duplicate IDs in '+str(path))
 return d.set_index('ID')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def values(data,cols):
 if not set(cols).issubset(data.columns):raise ValueError('Missing required features: '+str(sorted(set(cols)-set(data.columns))))
 x=data[cols].apply(pd.to_numeric,errors='coerce')
 if not np.isfinite(x.to_numpy()).all():raise ValueError('Nonfinite or missing model features')
 return x

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='operation',required=True)
 fit=sub.add_parser('freeze');infer=sub.add_parser('predict')
 for p in (fit,infer):
  p.add_argument('--clinical-model',required=True);p.add_argument('--traditional-model',required=True);p.add_argument('--clinical',required=True);p.add_argument('--traditional',required=True);p.add_argument('--state',required=True)
 fit.add_argument('--training-labels',required=True);infer.add_argument('--output',required=True)
 a=ap.parse_args();c=read(a.clinical);t=read(a.traditional);cm=joblib.load(a.clinical_model);tm=joblib.load(a.traditional_model);models={'clinical':cm,'traditional':tm};data={'clinical':c,'traditional':t};paths={'clinical':a.clinical_model,'traditional':a.traditional_model}
 if a.operation=='freeze':
  labs=read(a.training_labels);train=labs.index[labs.group=='train'];
  if len(train)==0:raise ValueError('No development cases')
  state={'schema':'ohp_standalone_tabular_v1','train_n':len(train),'branches':{}}
  for k,m in models.items():
   if not set(train).issubset(data[k].index):raise ValueError('Development cases missing for '+k)
   cols=list(m.feature_names_in_);x=values(data[k].loc[train],cols);sd=x.std(axis=0,ddof=1)
   if (sd<=0).any() or not np.isfinite(sd).all():raise ValueError('Invalid training scale')
   state['branches'][k]={'features':cols,'mean':x.mean(axis=0).to_dict(),'std':sd.to_dict(),'model_sha256':sha(paths[k])}
  if Path(a.state).exists():raise FileExistsError('Frozen state exists; choose a new path')
  Path(a.state).write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'operation':'freeze','train_n':len(train),'onekey_loaded':any(s=='onekey_algo' or s.startswith('onekey_algo.') for s in sys.modules)}));return
 state=json.loads(Path(a.state).read_text(encoding='utf8'))
 if state.get('schema')!='ohp_standalone_tabular_v1':raise ValueError('Invalid state')
 if set(c.index)!=set(t.index):raise ValueError('Clinical and traditional patient sets differ')
 ids=c.index;result=pd.DataFrame(index=ids)
 for k,m in models.items():
  s=state['branches'][k];cols=s['features']
  if sha(paths[k])!=s['model_sha256'] or list(m.feature_names_in_)!=cols:raise ValueError('Model differs from frozen state for '+k)
  x=values(data[k].loc[ids],cols);mu=pd.Series(s['mean']).loc[cols];sd=pd.Series(s['std']).loc[cols];result[k+'_probability']=m.predict_proba((x-mu)/sd)[:,1]
 if Path(a.output).exists():raise FileExistsError('Prediction output exists; choose a new path')
 result.index.name='ID';result.to_csv(a.output);print(json.dumps({'operation':'predict','n':len(ids),'onekey_loaded':any(s=='onekey_algo' or s.startswith('onekey_algo.') for s in sys.modules)}))
if __name__=='__main__':main()

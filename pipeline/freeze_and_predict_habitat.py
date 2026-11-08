# coding: utf-8
"""Habitat prediction using frozen training-donor KNN imputation; no OneKey import."""
import argparse,hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.impute import KNNImputer

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='operation',required=True);fit=sub.add_parser('freeze');pred=sub.add_parser('predict')
 fit.add_argument('--train-array',required=True);fit.add_argument('--feature-names',required=True);fit.add_argument('--model',required=True);fit.add_argument('--norm',required=True);fit.add_argument('--state',required=True)
 pred.add_argument('--features',required=True);pred.add_argument('--model',required=True);pred.add_argument('--state',required=True);pred.add_argument('--output',required=True)
 a=p.parse_args();m=joblib.load(a.model);selected=list(m.feature_names_in_)
 if a.operation=='freeze':
  tr=np.load(a.train_array)['train'];names=json.loads(Path(a.feature_names).read_text(encoding='utf8'));stats=json.loads(Path(a.norm).read_text(encoding='utf-8-sig'))
  if len(names)!=tr.shape[1] or len(names)!=len(set(names)) or not set(selected).issubset(names):raise ValueError('Invalid training feature schema')
  if np.isinf(tr).any():raise ValueError('Infinite training values')
  imp=KNNImputer(n_neighbors=5,weights='uniform').fit(tr)
  if not imp._valid_mask.all():raise ValueError('A training feature is entirely missing')
  state={'schema':'ohp_habitat_training_donor_knn_v1','features':names,'selected':selected,'mean':[float(stats[c]['mean']) for c in selected],'std':[float(stats[c]['std']) for c in selected],'model_sha256':sha(a.model),'donor_n':len(tr),'imputer':imp}
  if not np.isfinite(state['mean']).all() or not np.isfinite(state['std']).all() or np.any(np.asarray(state['std'])<=0):raise ValueError('Invalid normalization statistics')
  if Path(a.state).exists():raise FileExistsError('Frozen state exists; choose a new path')
  joblib.dump(state,a.state,compress=3);print(json.dumps({'operation':'freeze','donors':len(tr),'features':len(names),'onekey_loaded':any(x=='onekey_algo' or x.startswith('onekey_algo.') for x in sys.modules)}));return
 state=joblib.load(a.state)
 if state.get('schema')!='ohp_habitat_training_donor_knn_v1' or state['model_sha256']!=sha(a.model) or state['selected']!=selected:raise ValueError('Model and frozen state mismatch')
 d=pd.read_csv(a.features,dtype={'ID':str});cols=state['features'];
 if d.ID.duplicated().any() or not set(cols).issubset(d.columns):raise ValueError('Invalid target IDs or feature schema')
 raw=d[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
 if np.isinf(raw).any():raise ValueError('Infinite values')
 filled=state['imputer'].transform(raw)
 if filled.shape!=raw.shape or not np.isfinite(filled).all():raise ValueError('Incomplete imputation')
 if Path(a.output).exists():raise FileExistsError('Prediction output exists; choose a new path')
 idx=[cols.index(c) for c in selected];v=pd.DataFrame(filled[:,idx],columns=selected);x=(v-pd.Series(state['mean'],index=selected))/pd.Series(state['std'],index=selected);probs=m.predict_proba(x)[:,1];pd.DataFrame({'ID':d.ID,'habitat_probability':probs}).to_csv(a.output,index=False);print(json.dumps({'operation':'predict','cases':len(d),'missing_cells_filled':int(np.isnan(raw).sum()),'onekey_loaded':any(x=='onekey_algo' or x.startswith('onekey_algo.') for x in sys.modules)}))
if __name__=='__main__':main()

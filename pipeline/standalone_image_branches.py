# coding: utf-8
"""Independent 2D/2.5D inference on already prepared PNG/NPY; no OneKey import."""
import argparse,json,sys
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision import models,transforms

def model(path,dev):
 checkpoint=torch.load(path,map_location='cpu',weights_only=False);state=checkpoint.get('model_state_dict',checkpoint);state={str(k).removeprefix('module.'):v for k,v in state.items()};m=models.resnet34(weights=None,num_classes=2);m.load_state_dict(state,strict=True);return m.to(dev).eval()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cases',required=True);ap.add_argument('--weight2d',required=True);ap.add_argument('--weight25d');ap.add_argument('--only2d',action='store_true');ap.add_argument('--output',required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--batch',type=int,default=8);ap.add_argument('--tf32',action='store_true');a=ap.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=a.tf32;torch.backends.cudnn.allow_tf32=a.tf32;dev=torch.device(a.device)
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 if not a.only2d and not a.weight25d:raise ValueError('2.5D weight required')
 table=pd.read_csv(a.cases,dtype={'ID':str});required={'ID','png'} if a.only2d else {'ID','png','npy'}
 if not required.issubset(table) or table.ID.duplicated().any():raise ValueError('Invalid case table')
 tf=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);m2=model(a.weight2d,dev);m25=None if a.only2d else model(a.weight25d,dev);results=[]
 with torch.inference_mode():
  for i in range(0,len(table),a.batch):
   g=table.iloc[i:i+a.batch];x2=[];x25=[]
   for row in g.itertuples():
    x2.append(tf(np.asarray(Image.open(row.png).convert('RGB')).copy()))
    if not a.only2d:
     arr=np.load(row.npy)
     if arr.ndim==2:arr=np.stack([arr]*3,axis=-1)
     if arr.ndim!=3 or arr.shape[-1]!=3:raise ValueError('2.5D image shape must have 3 channels')
     x25.append(tf(np.ascontiguousarray(arr.astype(np.uint8))))
   p2=torch.softmax(m2(torch.stack(x2).to(dev)),1)[:,1].cpu().numpy()
   if a.only2d:results.extend((cid,float(v)) for cid,v in zip(g.ID,p2))
   else:
    p25=torch.softmax(m25(torch.stack(x25).to(dev)),1)[:,1].cpu().numpy();results.extend((cid,float(u),float(v)) for cid,u,v in zip(g.ID,p2,p25))
 from pathlib import Path
 if Path(a.output).exists():raise FileExistsError('Prediction output exists; choose a new path')
 cols=['ID','prob_2d'] if a.only2d else ['ID','prob_2d','prob_25d'];pd.DataFrame(results,columns=cols).to_csv(a.output,index=False);print(json.dumps({'cases':len(table),'device':str(dev),'only2d':a.only2d,'tf32':a.tf32,'batch':a.batch,'onekey_loaded':any(k=='onekey_algo' or k.startswith('onekey_algo.') for k in sys.modules)}))
if __name__=='__main__':main()

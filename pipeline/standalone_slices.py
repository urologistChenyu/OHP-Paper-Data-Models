# coding: utf-8
"""Independent reference-mask 2D/2.5D slice generation preserving image precision for windowing."""
import argparse,json,shutil,tempfile,sys
from pathlib import Path
import numpy as np,SimpleITK as sitk
from PIL import Image

def make_slices(image,mask):
 image=np.asarray(image);mask=np.asarray(mask)>0
 if image.shape!=mask.shape or image.ndim!=3:raise ValueError('Image/mask geometry mismatch')
 zs=np.flatnonzero(mask.reshape(mask.shape[0],-1).sum(1))
 if not len(zs):raise ValueError('Empty mask')
 z=int(max(zs,key=lambda k:int(mask[k].sum())));ys,xs=np.where(mask[z]);y0,y1,x0,x1=int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1
 def extract(zz,pad):
  a,b=max(y0-pad,0),min(y1+pad,mask.shape[1]);c,d=max(x0-pad,0),min(x1+pad,mask.shape[2]);v=np.clip((image[zz,a:b,c:d]+125.0)/300.0*255.0,0,255).astype(np.uint8);v[~mask[zz,a:b,c:d]]=0;return v
 d2=extract(z,0);layers=[]
 for zz in (z,z+1,z-1):
  zz=min(max(zz,0),image.shape[0]-1);layers.append(np.asarray(Image.fromarray(extract(zz,5)).resize((224,224),Image.Resampling.BICUBIC)))
 return d2,np.stack(layers,axis=-1).astype(np.uint8)

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--mask',required=True);p.add_argument('--output2d',required=True);p.add_argument('--output25d',required=True);p.add_argument('--scratch-root',required=True);a=p.parse_args();scratch=Path(a.scratch_root);scratch.mkdir(parents=True,exist_ok=True)
 if Path(a.output2d).exists() or Path(a.output25d).exists():raise FileExistsError('Slice output exists; choose new paths')
 with tempfile.TemporaryDirectory(dir=scratch,prefix='slices_') as d:
  ip=Path(d)/'image.nii.gz';mp=Path(d)/'mask.nii.gz';shutil.copyfile(a.image,ip);shutil.copyfile(a.mask,mp);im=sitk.ReadImage(str(ip));ms=sitk.ReadImage(str(mp))
  if im.GetSize()!=ms.GetSize() or not np.allclose(im.GetSpacing(),ms.GetSpacing()) or not np.allclose(im.GetOrigin(),ms.GetOrigin()) or not np.allclose(im.GetDirection(),ms.GetDirection()):raise ValueError('Image/mask spatial geometry mismatch')
  x,y=make_slices(sitk.GetArrayFromImage(im),sitk.GetArrayFromImage(ms));Path(a.output2d).parent.mkdir(parents=True,exist_ok=True);Path(a.output25d).parent.mkdir(parents=True,exist_ok=True);Image.fromarray(x).save(a.output2d);np.save(a.output25d,y)
 print(json.dumps({'shape2d':list(x.shape),'shape25d':list(y.shape),'onekey_loaded':any(k=='onekey_algo' or k.startswith('onekey_algo.') for k in sys.modules)}))
if __name__=='__main__':main()



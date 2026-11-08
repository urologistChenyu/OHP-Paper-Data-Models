# coding: utf-8
"""Independent image+reference-mask radiomics with the historic 1-D cluster centers.
No OneKey imports. Joint-development/validation cluster provenance must be disclosed.
"""
import argparse,json,shutil,tempfile,sys
from pathlib import Path
import numpy as np,pandas as pd,SimpleITK as sitk
from scipy import ndimage
from radiomics import featureextractor

def features(ex,image,mask,label):
 raw=ex.execute(image,mask,label=label)
 return {str(k).replace('-','_'):float(v) for k,v in raw.items() if not str(k).startswith('diagnostics')}

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--mask',required=True);p.add_argument('--id',required=True);p.add_argument('--yaml',required=True);p.add_argument('--centers',required=True);p.add_argument('--habitat-names',required=True);p.add_argument('--outdir',required=True);p.add_argument('--scratch-root',required=True);p.add_argument('--allow-missing-regions',action='store_true');p.add_argument('--skip-habitat',action='store_true');a=p.parse_args();dest=Path(a.outdir);dest.mkdir(parents=True,exist_ok=True);scratch=Path(a.scratch_root);scratch.mkdir(parents=True,exist_ok=True)
 if any((dest/x).exists() for x in ('traditional.csv','habitat.csv','radiomics_report.json')):raise FileExistsError('Radiomics output exists; choose a new directory')
 if not a.skip_habitat:
  centers=np.asarray(json.loads(Path(a.centers).read_text(encoding='utf8'))['centers'],dtype=float).reshape(-1);names=json.loads(Path(a.habitat_names).read_text(encoding='utf8'))
  if centers.shape!=(3,) or len(names)!=5502 or len(set(names))!=5502:raise ValueError('Invalid habitat model or feature schema')
 with tempfile.TemporaryDirectory(dir=scratch,prefix='radio_') as d:
  ip=Path(d)/'image.nii.gz';mp=Path(d)/'mask.nii.gz';yp=Path(d)/'settings.yaml';shutil.copyfile(a.image,ip);shutil.copyfile(a.mask,mp);shutil.copyfile(a.yaml,yp);im=sitk.ReadImage(str(ip));mask=sitk.ReadImage(str(mp))
  if im.GetSize()!=mask.GetSize() or not np.allclose(im.GetSpacing(),mask.GetSpacing()) or not np.allclose(im.GetOrigin(),mask.GetOrigin()) or not np.allclose(im.GetDirection(),mask.GetDirection()):raise ValueError('Image/mask geometry mismatch')
  ma=sitk.GetArrayFromImage(mask)>0
  if not ma.any():raise ValueError('Empty mask')
  padded=np.stack([ndimage.binary_dilation(z,np.ones((3,3),bool)) for z in ma]).astype(np.uint8);pad=sitk.GetImageFromArray(padded);pad.CopyInformation(mask)
  if not a.skip_habitat:
   image=sitk.GetArrayFromImage(im).astype(np.float64);roi=padded>0;z=(np.clip(image[roi],-125,175)+125)/300;labels=np.abs(z[:,None]-centers[None,:]).argmin(1).astype(np.uint8)+1;regions=np.zeros(roi.shape,dtype=np.uint8);regions[roi]=labels;hab=sitk.GetImageFromArray(regions);hab.CopyInformation(mask)
  traditional=featureextractor.RadiomicsFeatureExtractor(str(yp));traditional.settings['correctMask']=True;trad=features(traditional,im,pad,1);pd.DataFrame([{'ID':a.id,**{'intra_'+k:v for k,v in trad.items()}}]).to_csv(dest/'traditional.csv',index=False)
  if a.skip_habitat:
   report={'onekey_loaded':any(k=='onekey_algo' or k.startswith('onekey_algo.') for k in sys.modules),'habitat_skipped':True,'mask_mode':'supplied mask'};(dest/'radiomics_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report),flush=True);return
  vals={};errors={}
  for h in (1,2,3):
   try:
    ex=featureextractor.RadiomicsFeatureExtractor(str(yp));ex.settings['correctMask']=True;vals.update({k+'_h'+str(h):v for k,v in features(ex,im,hab,h).items()})
   except Exception as e:
    errors[str(h)]=str(e)
    if not a.allow_missing_regions:raise RuntimeError('Habitat region extraction failed: '+str(h)) from e
  unexpected=[k for k in vals if k not in names];result={'ID':a.id,**{k:vals.get(k,np.nan) for k in names}};pd.DataFrame([result]).to_csv(dest/'habitat.csv',index=False);report={'onekey_loaded':any(k=='onekey_algo' or k.startswith('onekey_algo.') for k in sys.modules),'region_errors':errors,'habitat_missing':sum(not np.isfinite(result[k]) for k in names),'unexpected_features':len(unexpected),'cluster_center_provenance':'historic joint 677-case fit, not independent external validation','mask_mode':'supplied reference mask'};(dest/'radiomics_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()




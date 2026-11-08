"""Four clinical inputs, five frozen branches, and arithmetic-mean OHP inference.

Adapter for the desktop platform. Individual inference is delegated to the
previously checked independent pipeline; this adapter never imports OneKey.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT=Path(__file__).resolve().parent
APP=ROOT/'pipeline'
CONFIG=ROOT/'configs/model_manifest.json'
from package_paths import load_config,scratch_root
FIELDS=['Age','Tumorsize','NEUT','LYMPH']
BRANCHES=['Clinical_prob','Traditional_prob','Habitat_prob','2D_prob','2.5D_prob']
THRESHOLD=0.3530539465287861
PROTOCOL='OHP_mean5_ACC_20261104'
IO_SCRATCH=scratch_root()/'platform_io'

def read_volume_image(path):
    """Stage non-ASCII paths because Windows ITK cannot reliably open them."""
    IO_SCRATCH.mkdir(parents=True,exist_ok=True)
    if isinstance(path,list):
        if all(str(p).isascii() for p in path):return sitk.ReadImage([str(p) for p in path])
        with tempfile.TemporaryDirectory(dir=IO_SCRATCH,prefix='series_') as tmp:
            files=[]
            for i,p in enumerate(path):
                dst=Path(tmp)/(str(i)+'.dcm');shutil.copyfile(p,dst);files.append(str(dst))
            return sitk.ReadImage(files)
    source=Path(path)
    if str(source).isascii():return sitk.ReadImage(str(source))
    with tempfile.TemporaryDirectory(dir=IO_SCRATCH,prefix='read_') as tmp:
        if source.suffix.lower() in ['.mhd','.nhdr']:
            stage=Path(tmp)/'image';shutil.copytree(source.parent,stage);target=stage/source.name
        else:
            suffix='.nii.gz' if source.name.lower().endswith('.nii.gz') else source.suffix;target=Path(tmp)/('image'+suffix);shutil.copyfile(source,target)
        return sitk.ReadImage(str(target))

def write_volume_image(image,path):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);IO_SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=IO_SCRATCH,prefix='write_') as tmp:
        suffix='.nii.gz' if target.name.lower().endswith('.nii.gz') else target.suffix
        staged=Path(tmp)/('image'+suffix);sitk.WriteImage(image,str(staged),True);shutil.copyfile(staged,target)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');os.replace(tmp,path)
def case_id(value):
    s=Path(str(value)).name
    return re.sub(r'(?i)\.(nii\.gz|nii|nrrd|mha|mhd|nii\.png|nii\.npy)$','',s)
def find_inputs(root,limit=None):
    root=Path(root)
    if root.suffix.lower()=='.json':
        z=json.loads(root.read_text(encoding='utf-8-sig'));rows=[(str(x['ID']),list(x['dicom_files'])) for x in z['cases']]
    elif root.is_file():rows=[(case_id(root),root)]
    elif root.is_dir():
        images=sorted(p for p in root.iterdir() if p.is_file() and p.name.lower().endswith(('.nii.gz','.nii','.nrrd','.mha','.mhd')))
        if images:rows=[(case_id(p),p) for p in images]
        else:
            rows=[]
            for folder in [root]+sorted(p for p in root.rglob('*') if p.is_dir()):
                ids=sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []
                if not ids:continue
                if len(ids)!=1:raise ValueError('Multiple DICOM series: select the intended series files explicitly: '+str(folder))
                rel=folder.relative_to(root);cid=case_id(root) if str(rel)=='.' else rel.parts[0]
                files=list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder),ids[0]));rows.append((cid,files))
    else:raise FileNotFoundError(root)
    if len({x[0] for x in rows})!=len(rows):raise ValueError('More than one series per patient: choose one scoring CT series')
    if limit is not None and limit<1:raise ValueError('Case limit must be positive')
    return rows[:limit] if limit else rows
def selected_dicom_manifest(files,path):
    import pydicom
    groups={};patients={}
    for name in files:
        h=pydicom.dcmread(name,stop_before_pixels=True,specific_tags=['SeriesInstanceUID','PatientID','Modality'])
        if h.get('Modality')!='CT':raise ValueError('Only CT DICOM data files can be selected')
        uid=str(h.SeriesInstanceUID);cid=str(h.get('PatientID','')).strip()
        if not cid:raise ValueError('DICOM PatientID missing; use an explicitly identified case directory')
        if cid in {'.','..'} or not re.fullmatch(r'[\w.-]+',cid):raise ValueError('PatientID contains unsupported path characters')
        groups.setdefault(uid,{'ID':cid,'dicom_files':[]})['dicom_files'].append(str(Path(name).resolve()))
        patients.setdefault(cid,set()).add(uid)
    if any(len(u)>1 for u in patients.values()):raise ValueError('Multiple CT series for one patient; select one series per patient')
    save(path,{'cases':list(groups.values()),'input_mode':'selected_DICOM_data_files'})

def prepare_ct(source,work,cfg):
    raw=work/'dicom_ct.nii.gz'
    if isinstance(source,list):
        listing=work/'selected_dicom_files.txt';listing.write_text('\n'.join(source),encoding='utf8')
        converter=ROOT/'dicom/dicom_files_to_nifti_geometry_v2.py'
        subprocess.run([cfg['python312'],str(converter),'--file-list',str(listing),'--output',str(raw),'--report',str(work/'dicom_conversion.json')],check=True)
        image=read_volume_image(raw)
    else:image=read_volume_image(source)
    if image.GetDimension()!=3:raise ValueError('A three-dimensional CT is required')
    size=[int(n*s) for n,s in zip(image.GetSize(),image.GetSpacing())]
    if min(size)<1:raise ValueError('Invalid CT spacing or size')
    processed=sitk.Resample(image,size,sitk.Transform(),sitk.sitkNearestNeighbor,image.GetOrigin(),(1.,1.,1.),image.GetDirection(),0,image.GetPixelID())
    processed=sitk.Clamp(processed,lowerBound=-125,upperBound=175)
    target=work/(work.name+'.nii.gz');write_volume_image(processed,target)
    return target,processed

def check_assets():
    cfg=load_config(CONFIG)
    local={'tabular_state':APP/'frozen_tabular_state.json','habitat_state':APP/'frozen_habitat_knn.joblib','habitat_names':APP/'habitat_feature_names.json'}
    for key,expected in cfg['asset_sha256'].items():
        path=local.get(key,Path(cfg.get(key,'')))
        if not path.is_file() or digest(path)!=expected:raise ValueError('Frozen asset differs: '+key)
    state=json.loads((APP/'frozen_tabular_state.json').read_text(encoding='utf8'))
    if state['branches']['clinical']['features']!=FIELDS:raise ValueError('Clinical feature contract differs')
    return cfg

def preflight(input_path,clinical_path,mask_dir=None,limit=None):
    found=find_inputs(Path(input_path),limit)
    if not found:raise ValueError('No input CT found')
    clinical=pd.read_csv(clinical_path,dtype={'ID':str,'patient_id':str})
    if 'ID' in clinical and 'patient_id' in clinical:raise ValueError('Use only one case ID column')
    clinical=clinical.rename(columns={'patient_id':'ID'})
    if 'ID' not in clinical:raise ValueError('Clinical table needs ID or patient_id')
    if clinical.ID.isna().any() or clinical.ID.duplicated().any():raise ValueError('Missing or duplicate clinical case IDs')
    clinical=clinical.set_index('ID');clinical[FIELDS]=clinical[FIELDS].apply(pd.to_numeric,errors='raise')
    if not set(cid for cid,_ in found)<=set(clinical.index):raise ValueError('Clinical IDs do not match CT case IDs')
    x=clinical.loc[[cid for cid,_ in found],FIELDS]
    if not np.isfinite(x.values).all():raise ValueError('Incomplete clinical values')
    if (x.Age<0).any() or (x.Age>120).any() or (x.Tumorsize<=0).any() or (x.NEUT<0).any() or (x.LYMPH<=0).any():raise ValueError('Invalid age, tumour size, or blood-cell counts; check units')
    masks={}
    if mask_dir is not None:
        folder=Path(mask_dir)
        if not folder.is_dir():raise ValueError('Supplied-mask directory is required')
        for path in folder.iterdir():
            if not path.is_file() or not path.name.lower().endswith(('.nii.gz','.nii','.nrrd','.mha','.mhd')):continue
            cid=case_id(path)
            if cid in masks:raise ValueError('Duplicate supplied masks for '+cid)
            masks[cid]=path
        if not set(cid for cid,_ in found)<=set(masks):raise ValueError('Missing supplied mask for: '+', '.join(cid for cid,_ in found if cid not in masks))
    return found,clinical,masks

def align_supplied_mask(path,geometry,destination):
    raw=read_volume_image(path)
    if raw.GetDimension()!=3:raise ValueError('Supplied mask must be three dimensional')
    values=np.unique(sitk.GetArrayFromImage(raw))
    if not np.isin(values,[0,1]).all():raise ValueError('Supply a binary tumour mask containing 0 and 1; do not include kidney or multiple labels')
    aligned=sitk.Resample(raw,geometry,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    voxels=int(np.count_nonzero(sitk.GetArrayFromImage(aligned)))
    if voxels==0:raise ValueError('Supplied mask is empty after spatial alignment; check CT-mask pairing and geometry')
    write_volume_image(aligned,destination)
    return {'source':str(path),'source_sha256':digest(path),'aligned_sha256':digest(destination),'aligned_voxels':voxels,'interpolation':'nearest neighbour in physical coordinates','binary_labels':[0,1]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--clinical');p.add_argument('--output');p.add_argument('--mask-dir');p.add_argument('--limit',type=int);p.add_argument('--self-test',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    cfg=check_assets()
    if a.self_test:
        print(json.dumps({'status':'asset_contract_pass','clinical_features':FIELDS,'fusion':'arithmetic_mean_of_five','weights':[.2]*5,'threshold':THRESHOLD,'onekey_imported':any(k.startswith('onekey') for k in sys.modules)}));return
    if not all([a.input,a.clinical]) or (not a.preflight_only and not a.output):p.error('--input --clinical and --output (for inference) are required')
    found,clinical,masks=preflight(a.input,a.clinical,a.mask_dir,a.limit)
    if a.preflight_only:
        print(json.dumps({'status':'input_contract_pass','cases':[cid for cid,_ in found],'clinical_fields':FIELDS,'mask_mode':'supplied' if a.mask_dir else 'automatic','matched_masks':len([cid for cid,_ in found if cid in masks]),'fusion':'mean5','threshold':THRESHOLD},ensure_ascii=False));return
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():raise FileExistsError('Completed output exists; choose another output directory')
    workroot=out/'cases';workroot.mkdir(exist_ok=True);seg=[];result=[];status=[];start=time.time()
    def progress(state):save(out/'segmentation_progress.json',{'status':state,'cases':len(found),'segmentation':seg,'work_dir':str(workroot)})
    for cid,source in found:
        if cid in {'.','..'} or not re.fullmatch(r'[\w.-]+',cid):raise ValueError('Invalid case ID')
        work=workroot/cid;work.mkdir(exist_ok=True);prediction=work/'prediction'
        try:
            print('['+cid+'] 1% Preparing CT',flush=True);image,geometry=prepare_ct(source,work,cfg)
            clinical_path=work/'clinical.csv';clinical.loc[[cid],FIELDS].to_csv(clinical_path,index_label='ID')
            mask_args=['--auto-mask'];mask=prediction/'automatic_mask.nii.gz'
            if a.mask_dir:
                if cid not in masks:raise ValueError('No matched manual mask')
                mask=work/'supplied_mask.nii.gz';alignment=align_supplied_mask(masks[cid],geometry,mask);save(work/'mask_alignment.json',alignment);mask_args=['--mask',str(mask)]
            rec={'patient_id':cid,'mask':str(mask),'image':str(image),'mode':'supplied' if a.mask_dir else 'automatic'}
            if a.mask_dir:seg.append(rec);progress('processing')
            command=[cfg['python312'],str(APP/'run_reference_case.py'),'--image',str(image)]+mask_args+['--clinical',str(clinical_path),'--id',cid,'--fusion','mean5','--threshold',str(THRESHOLD),'--manifest',str(CONFIG),'--output',str(prediction)]
            logfile=work/'inference.log'
            with logfile.open('w',encoding='utf8') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                published=bool(a.mask_dir)
                while proc.poll() is None:
                    if not published and mask.exists():
                        try:
                            if np.any(sitk.GetArrayFromImage(read_volume_image(mask))>0):seg.append(rec);progress('processing');published=True;print('['+cid+'] 35% Segmentation available',flush=True)
                        except RuntimeError:pass
                    time.sleep(.5)
            if proc.returncode:raise RuntimeError('Inference failed; see case inference.log')
            z=pd.read_csv(prediction/'predictions.csv').iloc[0];r={'patient_id':cid}
            for dest,src in zip(BRANCHES,['clinical_probability','traditional_probability','habitat_probability','prob_2d','prob_25d']):r[dest]=float(z[src])
            r['fusion_prob']=float(np.mean([r[k] for k in BRANCHES]));r['OHP_percent']=r['fusion_prob']*100;r['pred_primary']=int(r['fusion_prob']>=THRESHOLD)
            if abs(r['fusion_prob']-float(z.fusion_probability))>1e-12:raise ValueError('Saved branches do not reconstruct the pipeline fusion')
            if not any(x['patient_id']==cid for x in seg):seg.append(rec)
            result.append(r);status.append({'patient_id':cid,'status':'success','mask_mode':rec['mode'],'message':'Review CT-mask overlay before use'})
            print('['+cid+'] 100% Complete',flush=True)
        except Exception as exc:
            status.append({'patient_id':cid,'status':'requires_review','mask_mode':'supplied' if a.mask_dir else 'automatic','message':'Supply a reviewed manual mask and rerun','detail':str(exc)})
            print('['+cid+'] requires review: '+str(exc),flush=True)
        pd.DataFrame(result,columns=['patient_id']+BRANCHES+['fusion_prob','OHP_percent','pred_primary']).to_csv(out/'patient_predictions.csv',index=False)
        pd.DataFrame(status).to_csv(out/'case_status.csv',index=False,encoding='utf-8-sig');progress('processing')
    manifest={'protocol':PROTOCOL,'batch_id':digest(out/'case_status.csv'),'cases':len(found),'successful_cases':len(result),'failed_cases':[r for r in status if r['status']!='success'],'clinical_features':FIELDS,'fusion':{'method':'mean5','weights':[.2]*5,'weights_order':BRANCHES,'threshold':THRESHOLD},'work_dir':str(workroot),'segmentation':seg,'model_asset_sha256':cfg['asset_sha256'],'mask_mode':'supplied' if a.mask_dir else 'automatic','elapsed_seconds':time.time()-start,'onekey_imported':any(k.startswith('onekey') for k in sys.modules),'pipeline':str(APP/'run_reference_case.py'),'pipeline_sha256':digest(APP/'run_reference_case.py')}
    save(out/'run_manifest.json',manifest);progress('complete');print('Completed: '+str(out),flush=True)
    if not result:raise RuntimeError('No OHP scores emitted; review the failed-case list')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);raise

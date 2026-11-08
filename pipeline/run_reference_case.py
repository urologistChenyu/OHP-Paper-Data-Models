# coding: utf-8
"""Research prototype: CT with a supplied or automatically generated mask to branch probabilities.
No OneKey imports. Failed automatic-mask quality checks stop downstream inference.
"""
import argparse,hashlib,json,subprocess,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from package_paths import load_config
import pandas as pd

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--mask');g.add_argument('--auto-mask',action='store_true');p.add_argument('--clinical',required=True);p.add_argument('--id',required=True);p.add_argument('--fusion',choices=['mean5','median5','median4_no25','mean3_nohab','median3_clin_hab_2d'],required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--threshold',type=float);p.add_argument('--labels');a=p.parse_args();root=Path(__file__).resolve().parent;cfg=load_config(a.manifest);out=Path(a.output);
 if cfg.get('image_device')=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
 paths={k:Path(cfg[k]) for k in ('radiomics_yaml','clinical_model','traditional_model','weight2d')}
 paths['tabular_state']=root/'frozen_tabular_state.json'
 if a.fusion!='mean3_nohab':
  paths.update({k:Path(cfg[k]) for k in ('cluster_centers','habitat_model','weight25d')})
  paths['habitat_state']=root/'frozen_habitat_knn.joblib';paths['habitat_names']=root/'habitat_feature_names.json'
 for key,path in paths.items():
  if key not in cfg.get('asset_sha256',{}) or not path.is_file() or digest(path)!=cfg['asset_sha256'][key]:raise ValueError('Asset hash mismatch or missing: '+key)
 if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory is not empty; choose a new directory')
 out.mkdir(parents=True,exist_ok=True);scratch=Path(cfg['scratch_root']);scratch.mkdir(parents=True,exist_ok=True);records=[];started=time.time()
 def run(stage,python,script,args):
  cmd=[str(python),str(root/script)]+list(map(str,args));r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',env=dict(os.environ,PYTHONIOENCODING='utf-8'));records.append({'stage':stage,'exit_code':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr[-4000:]});(out/'run_progress.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
  if r.returncode!=0:
   quality_path=out/'auto_mask_report.json'
   if stage=='auto_mask' and quality_path.is_file():
    quality=json.loads(quality_path.read_text(encoding='utf8'))
    if quality.get('status')=='requires_manual_mask':
     (out/'case_status.json').write_text(json.dumps({'ID':a.id,'status':'requires_manual_mask','ohp_probability':None,'reason':quality.get('reason'),'message':quality.get('recommendation'),'downstream_inference':'not_run'},ensure_ascii=False,indent=2),encoding='utf8')
   raise RuntimeError(stage+' failed; see run_progress.json')
  print(stage,'done',flush=True)
 mask_path=a.mask
 if a.auto_mask:
  mask_path=out/'automatic_mask.nii.gz'
  run('auto_mask',cfg['python312'],'standalone_auto_mask.py',['--image',a.image,'--output',mask_path,'--report',out/'auto_mask_report.json','--source',cfg['segmentation_source'],'--model-root',cfg['segmentation_model_root'],'--scratch-root',scratch/'segmentation'])
 s2=out/'slice_2d.png';s25=out/'slice_25d.npy';run('slice',cfg['python312'],'standalone_slices.py',['--image',a.image,'--mask',mask_path,'--output2d',s2,'--output25d',s25,'--scratch-root',scratch])
 nohab=a.fusion=='mean3_nohab'
 rad=out/'radiomics';run('radiomics',cfg['python39'],'standalone_radiomics.py',['--image',a.image,'--mask',mask_path,'--id',a.id,'--yaml',cfg['radiomics_yaml'],'--centers',cfg['cluster_centers'],'--habitat-names',root/'habitat_feature_names.json','--outdir',rad,'--scratch-root',scratch]+(['--skip-habitat'] if nohab else ['--allow-missing-regions']))
 tab=out/'tabular_predictions.csv';run('tabular',cfg['python39'],'freeze_and_predict.py',['predict','--clinical-model',cfg['clinical_model'],'--traditional-model',cfg['traditional_model'],'--clinical',a.clinical,'--traditional',rad/'traditional.csv','--state',root/'frozen_tabular_state.json','--output',tab])
 hab=out/'habitat_predictions.csv'
 if not nohab:run('habitat',cfg['python39'],'freeze_and_predict_habitat.py',['predict','--features',rad/'habitat.csv','--model',cfg['habitat_model'],'--state',root/'frozen_habitat_knn.joblib','--output',hab])
 cases=out/'image_cases.csv';pd.DataFrame([{'ID':a.id,'png':str(s2),'npy':str(s25)}]).to_csv(cases,index=False);img=out/'image_predictions.csv';deep_args=['--cases',cases,'--weight2d',cfg['weight2d'],'--output',img,'--device',cfg['image_device'],'--batch','1']+(['--only2d'] if nohab else ['--weight25d',cfg['weight25d']])+(['--tf32'] if cfg.get('image_tf32') else []);run('deep',cfg['python312'],'standalone_image_branches.py',deep_args)
 args=['--tabular',tab,'--image',img,'--method',a.fusion,'--output',out/'predictions.csv','--metrics',out/'metrics.json']+([] if nohab else ['--habitat',hab])
 if a.threshold is not None:args+=['--threshold',str(a.threshold)]
 if a.labels:args+=['--labels',a.labels]
 run('fusion',cfg['python39'],'fuse_and_evaluate.py',args)
 manifest={'scope':'one research case; mask mode recorded separately','fusion':a.fusion,'image_sha256':digest(a.image),'mask_sha256':digest(mask_path),'mask_mode':'automatic' if a.auto_mask else 'supplied','clinical_sha256':digest(a.clinical),'elapsed_seconds':round(time.time()-started,1),'image_device':cfg['image_device'],'image_tf32':cfg.get('image_tf32',False),'asset_sha256':{k:cfg['asset_sha256'][k] for k in paths},'pipeline_sha256':{name:digest(root/name) for name in ('run_reference_case.py','standalone_slices.py','standalone_radiomics.py','freeze_and_predict.py','freeze_and_predict_habitat.py','standalone_image_branches.py','fuse_and_evaluate.py')},'runtime_sha256':{name:digest(cfg[name]) for name in ('python39','python312')},'habitat_warning':None if nohab else 'historic 677-case unsupervised clustering included validation images','licenses':'not yet fully audited','stages':records};(out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'status':'done','elapsed_seconds':manifest['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()







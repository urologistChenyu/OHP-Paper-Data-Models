# coding: utf-8
"""Independent automatic mask using the existing KiTS21 nnU-Net assets.
Intermediate and trainer outputs are redirected to the requested scratch directory.
"""
import argparse,gc,json,os,pickle,sys,time
from pathlib import Path
import numpy as np

def mask_quality(mask,expected_shape):
 shape_matches=tuple(mask.shape)==tuple(expected_shape);voxels=int(np.count_nonzero(mask))
 reason='shape_mismatch' if not shape_matches else 'empty_tumour_mask' if voxels==0 else None
 return {'status':'requires_manual_mask' if reason else 'ready','reason':reason,'mask_shape':list(mask.shape),'expected_shape':list(expected_shape),'tumour_voxels':voxels,'recommendation':'Automatic segmentation did not pass quality checks. Review or delineate the tumour, then resubmit through the supplied-mask route.' if reason else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);p.add_argument('--source',required=True);p.add_argument('--model-root',required=True);p.add_argument('--scratch-root',required=True);a=p.parse_args();scratch=Path(a.scratch_root);scratch.mkdir(parents=True,exist_ok=True);os.environ['KIDNEYAUTOSEG_TMP']=str(scratch);os.environ['TEMP']=str(scratch);os.environ['TMP']=str(scratch);sys.path.insert(0,str(a.source));import SimpleITK as sitk,torch
 from kidneyautoseg.io import read_volume,write_mask,ascii_tempdir
 from kidneyautoseg.kits_inference import verify_artifacts
 for name,value in [('int',int),('float',float),('product',np.prod)]:
  if name not in np.__dict__:setattr(np,name,value)
 from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
 from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
 root=Path(a.model_root);prov=verify_artifacts(root);volume=read_volume(Path(a.image));torch.set_num_threads(4);use_cuda=bool(torch.cuda.is_available());started=time.time();trainout=scratch/'trainer_output';preprocessed=scratch/'preprocessed';trainout.mkdir(parents=True,exist_ok=True);preprocessed.mkdir(parents=True,exist_ok=True)
 with (root/'fold_0/model_final_checkpoint.model.pkl').open('rb') as f:info=pickle.load(f)
 args=list(info['init']);args[0]=str(root/'plans.pkl');args[1]=0;args[2]=str(trainout);args[3]=str(preprocessed);trainer=nnUNetTrainerV2(*args);trainer.process_plans(info['plans']);trainer.initialize(training=False);ck=torch.load(root/'fold_0/model_final_checkpoint.model',map_location='cpu',weights_only=False);trainer.load_checkpoint_ram(ck,train=False);del ck
 try:
  with ascii_tempdir() as d:
   temp=Path(d);ct=temp/'ct.nii.gz';fl=sitk.GetImageFromArray(volume.array.astype(np.float32,copy=False));fl.CopyInformation(volume.image);sitk.WriteImage(fl,str(ct),True);print('AUTO_PREPROCESS',flush=True);data,_,props=trainer.preprocess_patient([str(ct)]);print('AUTO_INFERENCE',flush=True);_,prob=trainer.predict_preprocessed_data_return_seg_and_softmax(data,do_mirroring=True,mirror_axes=trainer.data_aug_params['mirror_axes'],use_sliding_window=True,step_size=0.5,use_gaussian=True,all_in_gpu=False,mixed_precision=use_cuda,verbose=False);prob=prob.transpose([0]+[i+1 for i in trainer.transpose_backward]);seg=temp/'labels.nii.gz';save_segmentation_nifti_from_softmax(prob,str(seg),props,order=1,verbose=False);labels=sitk.GetArrayFromImage(sitk.ReadImage(str(seg)));mask=(labels==2).astype(np.uint8)
  quality=mask_quality(mask,volume.array.shape)
  if quality['status']!='ready':
   report={'engine':'KiTS21 nnU-Net 3d_fullres fold 0',**quality,'cuda':use_cuda,'seconds':round(time.time()-started,1),'onekey_loaded':any(k=='onekey_algo' or k.startswith('onekey_algo.') for k in sys.modules),'requires_human_review':True,'ohp_probability':None,'model_source':prov.get('source')};report_path=Path(a.report);report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False),flush=True);raise RuntimeError(quality['reason']+'; see automatic mask quality report')
  out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);write_mask(mask,volume.image,out);report={'engine':'KiTS21 nnU-Net 3d_fullres fold 0','tumour_voxels':int(mask.sum()),'cuda':use_cuda,'seconds':round(time.time()-started,1),'onekey_loaded':any(k=='onekey_algo' or k.startswith('onekey_algo.') for k in sys.modules),'requires_human_review':True,'clinical_release_approved':prov.get('clinical_release_approved',False),'model_source':prov.get('source')};Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False),flush=True)
 finally:
  del trainer;gc.collect()
  if use_cuda:torch.cuda.empty_cache()
if __name__=='__main__':main()

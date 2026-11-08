"""Relocatable configuration; computation and trained parameters are unchanged."""
from pathlib import Path
import json,os,sys,tempfile

ROOT=Path(__file__).resolve().parent

def python_for(kind):
    variable='OHP_PYTHON39' if kind=='39' else 'OHP_PYTHON312'
    override=os.environ.get(variable)
    if override:return Path(override).expanduser().resolve()
    portable=ROOT/'runtime'/('python39' if kind=='39' else 'python312')/('python.exe' if os.name=='nt' else 'bin/python3')
    if portable.is_file():return portable
    folder=ROOT/('.venv39' if kind=='39' else '.venv312')
    candidate=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if candidate.is_file():return candidate
    target=(3,9) if kind=='39' else (3,12)
    if sys.version_info[:2]==target:return Path(sys.executable)
    return candidate

def scratch_root():
    chosen=os.environ.get('OHP_WORK_DIR')
    if chosen:return Path(chosen).expanduser().resolve()
    if str(ROOT).isascii():return ROOT/'work'
    # Windows ITK requires an ASCII intermediate path, including for Chinese installations.
    if os.name=='nt':return Path(ROOT.anchor)/'OHP_work'
    return Path(tempfile.gettempdir())/'ohp_work'

def load_config(path=None):
    cfg=json.loads(Path(path or ROOT/'configs/model_manifest.json').read_text(encoding='utf-8'))
    for key in ['radiomics_yaml','cluster_centers','clinical_model','traditional_model','habitat_model',
                'weight2d','weight25d','segmentation_source','segmentation_model_root']:
        cfg[key]=str((ROOT/cfg[key]).resolve())
    cfg['python39']=str(python_for('39'));cfg['python312']=str(python_for('312'))
    cfg['scratch_root']=str(scratch_root())
    if os.environ.get('OHP_DEVICE'):cfg['image_device']=os.environ['OHP_DEVICE']
    elif cfg.get('image_device')=='cuda':
        try:
            import torch
            if not torch.cuda.is_available():cfg['image_device']='cpu'
        except ImportError:pass
    return cfg

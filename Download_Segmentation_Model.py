"""Download the fixed KiTS21 segmentation checkpoint and verify its SHA256."""
from pathlib import Path
import urllib.request,hashlib,os
URL='https://github.com/urologistChenyu/OHP-Paper-Data-Models/releases/download/v1.0.0/KiTS21_fold0_model_final_checkpoint.model'
EXPECTED='d64a21c10973c459870297e57e39811304c689e3b9bddd5bbaeb7a8384d64cf7'
def main():
    destination=Path(__file__).resolve().parent/'models/kits21_fold0/fold_0/model_final_checkpoint.model'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_file():
        with destination.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest==EXPECTED:print('The segmentation checkpoint is already complete.');return
    temporary=destination.with_suffix('.download')
    request=urllib.request.Request(URL,headers={'User-Agent':'OHP-model-download'})
    h=hashlib.sha256()
    with urllib.request.urlopen(request,timeout=180) as response,temporary.open('wb') as out:
        while block:=response.read(1024*1024):out.write(block);h.update(block)
    if h.hexdigest()!=EXPECTED:raise RuntimeError('Checkpoint checksum mismatch. Download again before use.')
    os.replace(temporary,destination);print('Segmentation checkpoint downloaded and verified.')
if __name__=='__main__':main()

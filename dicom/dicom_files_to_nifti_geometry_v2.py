"""Convert selected DICOM files from one CT series to NIfTI; research validation utility."""
import argparse,json,shutil,tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from package_paths import scratch_root
import numpy as np,pydicom,SimpleITK as sitk

p=argparse.ArgumentParser();p.add_argument('--file-list',required=True,help='UTF-8 text file listing selected DICOM files, one per line');p.add_argument('--output',required=True);p.add_argument('--report',required=True);a=p.parse_args()
paths=[Path(v.strip()) for v in Path(a.file_list).read_text(encoding='utf-8-sig').splitlines() if v.strip()]
if not paths or len(paths)!=len(set(paths)) or not all(x.is_file() for x in paths):raise ValueError('Missing or duplicate input files')
if Path(a.output).exists() or Path(a.report).exists():raise FileExistsError('Choose new output and report paths')
headers=[pydicom.dcmread(str(x),stop_before_pixels=True) for x in paths]
if any(str(getattr(h,'Modality',''))!='CT' for h in headers):raise ValueError('Every selected file must be a CT object')
if any(not getattr(h,'SeriesInstanceUID',None) for h in headers) or len({str(h.SeriesInstanceUID) for h in headers})!=1:raise ValueError('Select files from exactly one CT series')
if len(paths)==1:
 frames=int(getattr(headers[0],'NumberOfFrames',1))
 if frames<2:raise ValueError('One ordinary DICOM file is one slice; select the full CT series as multiple data files')
 im=sitk.ReadImage(str(paths[0]));ordering='single_multiframe';dz=None
else:
 if any(int(getattr(h,'NumberOfFrames',1))!=1 for h in headers):raise ValueError('Do not mix multiframe and single-frame objects')
 if any(not hasattr(h,'ImageOrientationPatient') or not hasattr(h,'ImagePositionPatient') for h in headers):raise ValueError('Slice geometry tags are missing')
 orient=np.asarray(headers[0].ImageOrientationPatient,dtype=float)
 if not np.isfinite(orient).all() or not np.isclose(np.linalg.norm(orient[:3]),1.,atol=1e-4) or not np.isclose(np.linalg.norm(orient[3:]),1.,atol=1e-4) or not np.isclose(np.dot(orient[:3],orient[3:]),0.,atol=1e-4):raise ValueError('Invalid orientation basis')
 n=np.cross(orient[:3],orient[3:]);n=n/np.linalg.norm(n)
 for h in headers:
  if not np.allclose(np.asarray(h.ImageOrientationPatient,dtype=float),orient,atol=1e-4):raise ValueError('Inconsistent image orientation')
  if (int(h.Rows),int(h.Columns))!=(int(headers[0].Rows),int(headers[0].Columns)) or not np.allclose(h.PixelSpacing,headers[0].PixelSpacing,atol=1e-4):raise ValueError('Inconsistent matrix or in-plane spacing')
 coords=np.asarray([np.asarray(h.ImagePositionPatient,dtype=float) for h in headers]);positions=coords@n;order=np.argsort(positions);spacing=np.diff(positions[order]);
 expected_coords=coords[order[0]]+(positions[order]-positions[order[0]])[:,None]*n[None,:]
 if not np.allclose(coords[order],expected_coords,rtol=0,atol=0.1):raise ValueError('In-plane slice positions drift; inspect gantry tilt or mixed acquisition')
 if np.any(spacing<1e-4):raise ValueError('Duplicate or non-distinct slice positions')
 if len(spacing)>1 and not np.allclose(spacing,np.median(spacing),rtol=0.02,atol=0.1):raise ValueError('Irregular slice spacing; inspect acquisition before use')
 reader=sitk.ImageSeriesReader();reader.SetFileNames([str(paths[i]) for i in order]);im=reader.Execute();ordering='orientation_and_position';dz=float(np.median(spacing))
 # Independently verify slice data ordering before restoring direction from IOP
 # and the sorted physical positions. Some scanners report negative
 # SpacingBetweenSlices although the submitted file list is sorted ascending.
 volume=sitk.GetArrayViewFromImage(im)
 for k in (0,len(order)-1):
  obj=pydicom.dcmread(str(paths[order[k]]));expected=pydicom.pixels.apply_modality_lut(obj.pixel_array,obj)
  if expected.shape!=volume[k].shape or not np.allclose(expected,volume[k],rtol=0,atol=0.01):raise ValueError('DICOM pixel order or modality transform differs from assembled volume')
 direction=np.column_stack((orient[:3],orient[3:],n))
 im.SetDirection(tuple(float(v) for v in direction.ravel()));im.SetOrigin(tuple(float(v) for v in headers[order[0]].ImagePositionPatient));im.SetSpacing((float(headers[0].PixelSpacing[1]),float(headers[0].PixelSpacing[0]),dz))
if im.GetDimension()!=3 or im.GetSize()[2]<2:raise ValueError('Expected a 3D CT volume with at least two slices')
if len(paths)>1 and im.GetSize()[2]!=len(paths):raise ValueError('Output slice count mismatch')
if any(not np.isfinite(x) or x<=0 for x in im.GetSpacing()):raise ValueError('Invalid spacing')
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
scratch=scratch_root()/'dicom';scratch.mkdir(parents=True,exist_ok=True)
tmp=tempfile.TemporaryDirectory(dir=scratch,prefix='ct_'); temporary=Path(tmp.name)/'ct.nii.gz';sitk.WriteImage(im,str(temporary),True)
check=sitk.ReadImage(str(temporary))
if check.GetSize()!=im.GetSize() or not np.allclose(check.GetSpacing(),im.GetSpacing(),atol=1e-4) or not np.allclose(check.GetOrigin(),im.GetOrigin(),atol=1e-3) or not np.allclose(check.GetDirection(),im.GetDirection(),atol=1e-5) or not np.array_equal(sitk.GetArrayFromImage(check),sitk.GetArrayFromImage(im)):raise RuntimeError('NIfTI roundtrip voxel or spatial metadata mismatch')
shutil.copyfile(temporary,a.output);tmp.cleanup()
report={'scope':'research conversion of selected DICOM CT files; not a validated medical device','input_files':len(paths),'ordering':ordering,'direction_basis':('reader_multiframe' if len(paths)==1 else 'IOP plus sorted physical positions, pixel order verified'),'image_size':list(im.GetSize()),'spacing_mm':list(im.GetSpacing()),'median_slice_separation_mm':dz,'roundtrip_voxels_and_geometry_match':True,'cautions':['confirm the selected files form the complete intended acquisition','compare CT and mask geometry and coordinate systems before inference','no OHP prediction or automatic segmentation assessed']}
Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report),flush=True)






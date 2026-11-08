"""Frozen KiTS21 nnU-Net baseline. No annotation or historical-case access at runtime."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Callable
import hashlib
import json
import gc
import numpy as np

from .io import Volume, ascii_tempdir


def verify_artifacts(root: Path) -> dict:
    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    for relative, expected in provenance['files'].items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f'Missing model file: {relative}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f'Model checksum failed: {relative}')
    return provenance


@dataclass
class Prediction:
    mask: np.ndarray
    kidneys: np.ndarray
    details: dict


def predict(volume: Volume, model_root: Path, progress: Callable) -> Prediction:
    import SimpleITK as sitk
    import torch
    import pickle
    # This old, published nnU-Net architecture predates NumPy 2.
    for name, value in [('int', int), ('float', float), ('product', np.prod)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    use_cuda = bool(torch.cuda.is_available())
    torch.set_num_threads(4)
    from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
    root = Path(model_root)
    with (root / 'fold_0/model_final_checkpoint.model.pkl').open('rb') as stream:
        info = pickle.load(stream)
    # Reconstruct from the distributed plans, replacing the original author's paths.
    args = list(info['init'])
    args[0] = str(root / 'plans.pkl')
    args[1] = 0
    args[2] = str(root)
    preprocessed = root / 'preprocessed'
    preprocessed.mkdir(parents=True, exist_ok=True)
    args[3] = str(preprocessed)
    trainer = nnUNetTrainerV2(*args)
    trainer.process_plans(info['plans'])
    trainer.initialize(training=False)
    checkpoint = torch.load(root / 'fold_0/model_final_checkpoint.model', map_location='cpu', weights_only=False)
    trainer.load_checkpoint_ram(checkpoint, train=False)
    del checkpoint
    try:
        with ascii_tempdir() as directory:
            tmp = Path(directory)
            input_path = tmp / 'ct.nii.gz'
            # nnU-Net's legacy preprocessor otherwise promotes int32 CT values and
            # needs a second 250 MB working array on large studies.
            float_image = sitk.GetImageFromArray(volume.array.astype(np.float32, copy=False))
            float_image.CopyInformation(volume.image)
            sitk.WriteImage(float_image, str(input_path), True)
            progress('Kidney and tumour segmentation: preprocessing CT', 0.22)
            data, _, properties = trainer.preprocess_patient([str(input_path)])
            progress('Kidney and tumour segmentation: 3D inference', 0.35)
            _, probabilities = trainer.predict_preprocessed_data_return_seg_and_softmax(
                data, do_mirroring=True, mirror_axes=trainer.data_aug_params['mirror_axes'],
                use_sliding_window=True, step_size=0.5, use_gaussian=True,
                all_in_gpu=False, mixed_precision=use_cuda, verbose=False)
            tumour_probability = probabilities[2]
            predicted_labels = probabilities.argmax(0)
            p = tumour_probability[predicted_labels == 2]
            details = {'engine': 'KiTS21 nnU-Net 3d_fullres fold 0',
                       'uses_ground_truth': False, 'trained_on_user_cases': False,
                       'tumour_probability_mean': float(p.mean()) if p.size else 0.0,
                       'note': 'Softmax values are not calibrated clinical confidence.'}
            probabilities = probabilities.transpose([0] + [i+1 for i in trainer.transpose_backward])
            progress('Restoring the original CT geometry', 0.68)
            output_path = tmp / 'labels.nii.gz'
            save_segmentation_nifti_from_softmax(probabilities, str(output_path), properties,
                                                order=1, verbose=False)
            labels = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path)))
        if labels.shape != volume.array.shape:
            raise RuntimeError('Predicted mask dimensions do not match the CT')
        return Prediction((labels == 2).astype(np.uint8), labels > 0, details)
    finally:
        del trainer
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from contextlib import contextmanager
import shutil
import tempfile
import os

import numpy as np


class ImageIOError(RuntimeError):
    pass


MEDICAL_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")


def is_image_file(path: str | Path) -> bool:
    return Path(path).name.lower().endswith(MEDICAL_SUFFIXES)


def ascii_tempdir():
    # Use the configured ASCII scratch path for medical-image I/O.
    # The portable launcher controls this directory; no original research drive is required.
    root = Path(os.environ.get("KIDNEYAUTOSEG_TMP", tempfile.gettempdir()))
    if not str(root).isascii():
        root = Path(os.environ.get("PUBLIC", "C:/Users/Public")) / "KidneyAutoSegTemp"
        root.mkdir(parents=True, exist_ok=True)
    if not str(root).isascii():
        raise ImageIOError("Cannot find an ASCII temporary path for the image reader")
    return tempfile.TemporaryDirectory(prefix="kidneyautoseg_", dir=root)


@contextmanager
def staged_image(path: str | Path):
    """ITK on Windows cannot reliably use non-ASCII filenames."""
    source = Path(path).resolve()
    if str(source).isascii():
        yield source
        return
    with ascii_tempdir() as directory:
        stage = Path(directory)
        if source.is_dir():
            shutil.copytree(source, stage / "dicom")
            yield stage / "dicom"
        elif source.suffix.lower() in {".mhd", ".nhdr"}:
            # Detached headers refer to companion files; keep relative references intact.
            shutil.copytree(source.parent, stage / "image")
            yield stage / "image" / source.name
        else:
            suffix = next((s for s in MEDICAL_SUFFIXES if source.name.lower().endswith(s)), source.suffix)
            target = stage / ("image" + suffix)
            shutil.copyfile(source, target)
            yield target


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise ImageIOError("SimpleITK is required for CT and NIfTI/NRRD I/O") from exc
    return sitk


@dataclass
class Volume:
    image: Any
    source: Path

    @property
    def array(self) -> np.ndarray:
        return _sitk().GetArrayFromImage(self.image)


def read_volume(path: str | Path) -> Volume:
    source = Path(path)
    if not source.exists():
        raise ImageIOError(f"input does not exist: {source}")
    sitk = _sitk()
    try:
      with staged_image(source) as readable:
       if source.is_dir():
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(readable))
        if not series_ids:
            raise ImageIOError(f"no DICOM series found in: {source}")
        if len(series_ids) > 1:
            raise ImageIOError("The DICOM directory contains multiple series. Select one intended CT series.")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(readable), series_ids[0]))
        image = reader.Execute()
       else:
        image = sitk.ReadImage(str(readable))
       if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ImageIOError("A single-channel 3D CT volume is required; 2D, 4D and multichannel images are unsupported.")
       if any(v <= 0 or not np.isfinite(v) for v in image.GetSpacing()):
        raise ImageIOError("The CT voxel spacing is invalid.")
       return Volume(image, source)
    except Exception as exc:  # pragma: no cover - backend-specific error text
        if isinstance(exc, ImageIOError):
            raise
        raise ImageIOError(f"Could not read image: {source}; {exc}") from exc


def write_mask(mask: np.ndarray, reference: Any, destination: str | Path) -> Path:
    sitk = _sitk()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mask.ndim != 3:
        raise ImageIOError("mask must be a 3-D array")
    image = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    # Write to a temporary file, then atomically replace; never leave a partial Mask.
    with ascii_tempdir() as directory:
        stage = Path(directory) / "mask.nii.gz"
        sitk.WriteImage(image, str(stage), useCompression=True)
        pending = target.with_name(target.name + ".partial")
        try:
            shutil.copyfile(stage, pending)
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
    return target


"""Synthetic geometry fixtures test software only, never medical accuracy."""
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid
import pytest


@pytest.fixture
def dicom_series(tmp_path):
    def create(folder=None, series_uid=None, offsets=(0, 2, 4), orientation=(1, 0, 0, 0, 1, 0), shape=(20, 24), exposure=100, ctdi=10):
        folder = folder or tmp_path / generate_uid().split('.')[-1]
        folder.mkdir(parents=True, exist_ok=True)
        uid = series_uid or generate_uid()
        study, frame, event = generate_uid(), generate_uid(), generate_uid()
        normal = np.cross(orientation[:3], orientation[3:])
        for i, distance in enumerate(offsets):
            path = folder / f"slice_{len(offsets)-i:03}.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID = uid
            ds.StudyInstanceUID = study
            ds.FrameOfReferenceUID = frame
            ds.IrradiationEventUID = event
            ds.Modality = 'CT'; ds.PatientAge = '040Y'
            ds.ImagePositionPatient = list(normal * distance)
            ds.ImageOrientationPatient = list(orientation)
            ds.PixelSpacing = [1, 1.5]
            ds.Rows, ds.Columns = shape
            ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 1
            ds.RescaleSlope = 1; ds.RescaleIntercept = -1000 + i
            ds.KVP = 120; ds.Exposure = exposure
            if ctdi is not None: ds.CTDIvol = ctdi
            pixels = np.zeros(shape, dtype=np.int16)
            pixels[3:-3, 3:-3] = 1040
            ds.PixelData = pixels.tobytes()
            ds.save_as(path, enforce_file_format=True)
        return folder, uid
    return create

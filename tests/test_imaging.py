import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from ct_dose.imaging import load_scan, aligned_mask, write_mask


def test_oblique_geometry_order_and_per_slice_hu(dicom_series):
    folder, uid = dicom_series(orientation=(0, 1, 0, 0, 0, 1))
    scan = load_scan(folder)
    assert scan.series_uid == uid
    assert scan.image.GetSize() == (24, 20, 3)
    assert scan.image.GetSpacing() == (1.5, 1, 2)
    np.testing.assert_allclose(scan.array[:, 5, 5], [40, 41, 42])
    assert scan.image.TransformIndexToPhysicalPoint((0, 0, 2)) == (4, 0, 0)


def test_reject_mixed_series(dicom_series, tmp_path):
    folder, uid = dicom_series(tmp_path / 'a')
    dicom_series(tmp_path / 'b')
    with pytest.raises(ValueError, match='Multiple'): load_scan(tmp_path)
    assert load_scan(tmp_path, uid).series_uid == uid


@pytest.mark.parametrize('offsets', [(0, 2, 5), (0, 2, 2)])
def test_reject_irregular_or_duplicate_slices(dicom_series, offsets):
    folder, _ = dicom_series(offsets=offsets)
    with pytest.raises(ValueError, match='Duplicate|irregular'): load_scan(folder)


def test_reject_missing_calibration(dicom_series):
    folder, _ = dicom_series()
    path = next(folder.glob('*.dcm')); ds = pydicom.dcmread(path)
    del ds.RescaleIntercept; ds.save_as(path)
    with pytest.raises(ValueError, match='Missing'): load_scan(folder)


def test_mask_alignment_uses_physical_origin(tmp_path):
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    mask = np.zeros((8, 8, 8), np.uint8); mask[2, 2, 2] = 1
    shifted = sitk.GetImageFromArray(mask); shifted.SetOrigin((1, 0, 0))
    path = tmp_path / 'mask.nii.gz'; sitk.WriteImage(shifted, str(path))
    aligned = aligned_mask(path, image)
    assert aligned[2, 2, 3] and aligned.sum() == 1
    shifted.SetOrigin((100, 0, 0)); sitk.WriteImage(shifted, str(path))
    with pytest.raises(ValueError, match='overlap'): aligned_mask(path, image)


def test_dicom_nifti_roundtrip(dicom_series, tmp_path):
    folder, _ = dicom_series()
    scan = load_scan(folder)
    path = tmp_path / 'ct.nii.gz'; sitk.WriteImage(scan.image, str(path))
    loaded = load_scan(path)
    np.testing.assert_array_equal(scan.array, loaded.array)
    np.testing.assert_allclose(scan.image.GetDirection(), loaded.image.GetDirection())


def test_flipped_mask_direction_preserves_physical_location(tmp_path):
    reference = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    values = np.zeros((8, 8, 8), np.uint8); values[3, 3, 2] = 1
    flipped = sitk.GetImageFromArray(values)
    flipped.SetOrigin((7, 0, 0)); flipped.SetDirection((-1., 0., 0., 0., 1., 0., 0., 0., 1.))
    path = tmp_path / 'flipped.nii.gz'; sitk.WriteImage(flipped, str(path))
    aligned = aligned_mask(path, reference)
    assert aligned[3, 3, 5] and aligned.sum() == 1

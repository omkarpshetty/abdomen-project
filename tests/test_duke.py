"""Synthetic software fixtures only; no medical accuracy claims."""
import json
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from ct_dose.duke import raw_image, LABELS
from ct_dose.modeling import ORGAN_FEATURES, predict, train, sha256
from tests.test_modeling import training_files


def test_raw_endianness_geometry_and_size(tmp_path):
    source = np.array([[[-1000, 40], [200, -3024]], [[-10, 80], [0, 3071]]], dtype=np.int16)
    raw = tmp_path / 'test.raw'
    source.astype('>i2').tofile(raw)
    metadata = dict(Abdo_X_Pixel=2, Abdo_Y_Pixel=2, Abdo_Z_Pixel=2)
    metadata.update({'XY Pixel Size (mm/pixel)': .8, 'Z Pixel Size (mm/pixel)': 5})
    image = raw_image(raw, metadata)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), source)
    assert image.GetSpacing() == (.8, .8, 5.)
    assert image.TransformIndexToPhysicalPoint((0, 0, 0)) == (0., 0., 5.)
    assert image.TransformIndexToPhysicalPoint((0, 0, 1)) == (0., 0., 0.)
    raw.write_bytes(b'wrong size')
    with pytest.raises(ValueError, match='byte count'):
        raw_image(raw, metadata)


def test_label_identity_does_not_duplicate_bilateral_targets():
    assert len(set(LABELS.values())) == len(LABELS)
    assert 'kidney_left' not in LABELS and 'kidney_right' not in LABELS


def test_organ_only_schema_protocol_and_experimental_gates(training_files, tmp_path):
    data, path = training_files
    frame = pd.read_csv(data, dtype={'patient_id': str})
    frame.water_equivalent_diameter_cm = np.nan
    frame.to_csv(data, index=False)
    manifest = json.loads(path.read_text())
    manifest.update(features=ORGAN_FEATURES, protocol_profile='test_fixture', experimental_only=True,
                    segmentation_resolution='fast', segmentation_version='2.8.0', acquisition_scope={'ctdi_phantom_cm': 32}, data_sha256=sha256(data))
    path.write_text(json.dumps(manifest))
    output = tmp_path / 'model'
    meta = train(data, path, output)
    assert 'water_equivalent_diameter_cm' not in meta['bounds']
    assert 'unavailable' in meta['test_stratified']['size_group']
    assert meta['status'] == 'experimental_geometry_unverified'
    assert meta['baseline']['method'] == 'Training-only per-organ mean'
    frame = frame[frame.patient_id.isin(meta['splits']['train'])]
    with pytest.raises(ValueError, match='protocol_profile'): predict(frame, output)
    context = dict(protocol_profile='test_fixture', segmentation_resolution='fast', segmentation_version='2.8.0')
    with pytest.raises(ValueError, match='phantom'): predict(frame, output, context)
    context['ctdi_phantom_cm'] = 32
    with pytest.raises(ValueError, match='Experimental'): predict(frame, output, context)
    context['allow_experimental'] = True
    first, _ = predict(frame, output, context)
    second, _ = predict(frame, output, context)
    np.testing.assert_array_equal(first, second)
    corrected = frame.assign(source='expert_correction')
    with pytest.raises(ValueError, match='Mask provenance'): predict(corrected, output, context)
    context['segmentation_resolution'] = 'full'
    with pytest.raises(ValueError, match='segmentation_resolution'): predict(frame, output, context)


def test_mixed_protocol_training_rejected(training_files, tmp_path):
    data, path = training_files
    frame = pd.read_csv(data); frame.loc[0, 'protocol'] = 'different_exposure'
    frame.to_csv(data, index=False)
    manifest = json.loads(path.read_text()); manifest['data_sha256'] = sha256(data)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='multiple acquisition protocols'):
        train(data, path, tmp_path / 'model')


def test_prepare_whitespace_headers_missing_organs_and_checked_resume(tmp_path, monkeypatch):
    from ct_dose import duke
    root = tmp_path / 'source'
    images = root / 'Patient_Images'; (images / 'abdo').mkdir(parents=True)
    dose_dir = root / 'Organ_Dose/Abdo_Dose'; dose_dir.mkdir(parents=True)
    metadata = {'Patient number': 'pt001', 'Age(Yr)': 40, 'Abdo_image_name': 'pt001.raw',
                'Abdo_X_Pixel': 3, 'Abdo_Y_Pixel': 3, 'Abdo_Z_Pixel': 3,
                'XY Pixel Size (mm/pixel)': 1, 'Z Pixel Size (mm/pixel)': 2,
                'Abdominopelvic CTDIvol (mGy)': 7.03}
    pd.DataFrame([metadata]).to_excel(images / 'Patient_information.xlsx', index=False)
    np.zeros((3, 3, 3), dtype='>i2').tofile(images / 'abdo/pt001.raw')
    dose_row = {name.replace(' (mGy)', '   (mGy)'): 3. for name in LABELS.values()}
    pd.DataFrame([{'Patient': 'pt001', **dose_row}]).to_excel(dose_dir / 'Abdo_Fixed.xlsx', index=False)
    monkeypatch.setattr(duke, 'acquire', lambda directory: root)
    calls = []
    def annotation(scan, output, organs, fast):
        calls.append(True)
        (output / 'raw_masks').mkdir(parents=True)
        mask_path = output / 'raw_masks/liver.nii.gz'
        mask_path.write_bytes(b'synthetic cached mask for software test')
        return {'measurements': [dict(organ='liver', volume_cm3=20., mean_hu=40., std_hu=10., coverage='not_verified')],
                'organs': {'liver': {'status': 'present', 'input_sha256': duke.digest(mask_path)}}}
    monkeypatch.setattr(duke, 'annotate', annotation)
    destination = tmp_path / 'features'
    with pytest.raises(ValueError, match="Patient limit"):
        duke.prepare(root, destination, limit_patients=0)
    manifest = duke.prepare(root, destination, limit_patients=1)
    assert manifest["selected_patients"] == 1
    assert "pilot" in manifest["selection_policy"].lower()
    assert manifest['experimental_only']
    assert manifest['patient_audit'][0]['excluded_organs'] == sorted(set(LABELS) - {'liver'})
    frame = pd.read_csv(destination / 'training.csv')
    assert len(frame) == 1 and frame.iloc[0].dose_mGy == 3.
    assert manifest['data_sha256'] == sha256(destination / 'training.csv')
    duke.prepare(root, destination)
    assert len(calls) == 1
    (destination / 'patients/pt001/annotation/raw_masks/liver.nii.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed cached mask'):
        duke.prepare(root, destination)

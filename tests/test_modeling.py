"""Synthetic labels here exercise contracts only; they are not accuracy evidence."""
import json
import numpy as np
import pandas as pd
import pytest
from ct_dose.modeling import train, predict, sha256, validate_features


@pytest.fixture
def training_files(tmp_path):
    rows = []
    for patient in range(20):
        for organ, volume in [('liver', 1200), ('spleen', 200)]:
            rows.append(dict(patient_id=str(patient), acquisition_id='one', organ=organ,
                             volume_cm3=volume + patient, mean_hu=40 + patient, std_hu=10,
                             ctdivol_mGy=5 + patient, water_equivalent_diameter_cm=20 + patient,
                             kvp=120, protocol='test_fixture', dose_mGy=5 + patient))
    path = tmp_path / 'references.csv'; pd.DataFrame(rows).to_csv(path, index=False)
    manifest = dict(dataset_id='SYNTHETIC-SOFTWARE-TEST-ONLY', source_url='https://example.invalid/test-fixture',
                    license='test fixture', reference_method='measured', reference_evidence='SIMULATED MANIFEST FOR CONTRACT TESTING ONLY',
                    population='adult', region='abdomen_pelvis', data_sha256=sha256(path))
    mpath = tmp_path / 'manifest.json'; mpath.write_text(json.dumps(manifest))
    return path, mpath


def test_train_reload_disjoint_groups_and_deterministic_predictions(training_files, tmp_path):
    data, manifest = training_files
    output = tmp_path / 'model'; meta = train(data, manifest, output)
    groups = [set(meta['splits'][name]) for name in ['train', 'validation', 'test']]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    frame = pd.read_csv(data, dtype={'patient_id': str})
    frame = frame[frame.patient_id.isin(groups[0])]
    first, _ = predict(frame, output); second, _ = predict(frame, output)
    np.testing.assert_array_equal(first, second)
    assert np.ptp(first) > 0
    wrong = frame.copy(); wrong['organ'] = 'unsupported'
    with pytest.raises(ValueError, match='organs'): predict(wrong, output)
    modified = json.loads((output / 'manifest.json').read_text()); modified['features'] = ['wrong']
    (output / 'manifest.json').write_text(json.dumps(modified))
    with pytest.raises(ValueError, match='schema'): predict(frame, output)


def test_formula_labels_rejected(training_files, tmp_path):
    data, path = training_files
    manifest = json.loads(path.read_text()); manifest['reference_method'] = 'formula'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Formula'): train(data, path, tmp_path / 'model')


def test_missing_features_rejected(training_files):
    data, _ = training_files
    frame = pd.read_csv(data); frame.loc[0, 'ctdivol_mGy'] = np.nan
    with pytest.raises(ValueError, match='Missing'): validate_features(frame)


def test_checksum_and_runtime_checks(training_files, tmp_path):
    data, manifest = training_files; output = tmp_path / 'model'
    meta = train(data, manifest, output)
    frame = pd.read_csv(data)
    manifest_file = output / 'manifest.json'
    original = manifest_file.read_text()
    meta['dependencies']['scikit-learn'] = '0.0'
    manifest_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='runtime'): predict(frame, output)
    manifest_file.write_text(original)
    with open(output / 'model.joblib', 'ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'): predict(frame, output)


def test_out_of_range_predictions_withheld(training_files, tmp_path):
    data, manifest = training_files; output = tmp_path / 'model'
    train(data, manifest, output)
    frame = pd.read_csv(data); frame.ctdivol_mGy = 1000
    with pytest.raises(ValueError, match='training range'): predict(frame, output)

import math
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from ct_dose.imaging import load_scan
from ct_dose.dosimetry import dose_report, read_rdsr


def coded(value):
    ds = Dataset(); ds.CodingSchemeDesignator = 'DCM'; ds.CodeValue = value; ds.CodeMeaning = 'fixture'
    return ds


def rdsr(path, study, event_uid, dose=10, dlp=150, unit='mGy'):
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.88.67'; meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study
    event = Dataset(); event.ConceptNameCodeSequence = [coded('113819')]; event.ValueType = 'CONTAINER'
    uid = Dataset(); uid.ConceptNameCodeSequence = [coded('113769')]; uid.UID = event_uid; uid.ValueType = 'UIDREF'
    event.ContentSequence = [uid]
    for concept, value, units in [('113830', dose, unit), ('113838', dlp, 'mGy.cm')]:
        node = Dataset(); node.ConceptNameCodeSequence = [coded(concept)]; node.ValueType = 'NUM'
        measured = Dataset(); measured.NumericValue = value
        u = Dataset(); u.CodeValue = units; u.CodingSchemeDesignator = 'UCUM'; u.CodeMeaning = units
        measured.MeasurementUnitsCodeSequence = [u]; node.MeasuredValueSequence = [measured]
        event.ContentSequence.append(node)
    ds.ContentSequence = [event]; ds.save_as(path, enforce_file_format=True)
    return path


def test_missing_metadata_stays_missing(dicom_series):
    folder, _ = dicom_series(ctdi=None)
    result = dose_report(load_scan(folder))
    assert result['quantities']['ctdivol']['value'] is None
    assert result['quantities']['ssde']['value'] is None
    assert result['quantities']['effective_dose']['value'] is None
    assert 'total_patient_dose' not in result


def test_event_matching_and_deduplication(dicom_series, tmp_path):
    folder, _ = dicom_series(); scan = load_scan(folder); ds = scan.datasets[0]
    a = rdsr(tmp_path / 'a.dcm', ds.StudyInstanceUID, ds.IrradiationEventUID)
    b = rdsr(tmp_path / 'b.dcm', ds.StudyInstanceUID, ds.IrradiationEventUID)
    result = dose_report(scan, [a, b])
    assert len(result['irradiation_events']) == 1
    assert result['quantities']['dlp']['value'] == 150
    assert result['provided_events_dlp_sum']['value'] == 150
    rdsr(b, ds.StudyInstanceUID, ds.IrradiationEventUID, dose=11)
    with pytest.raises(ValueError, match='Conflicting'): read_rdsr([a, b])


def test_unmatched_rdsr_not_assigned(dicom_series, tmp_path):
    folder, _ = dicom_series(); scan = load_scan(folder)
    path = rdsr(tmp_path / 'sr.dcm', generate_uid(), generate_uid())
    result = dose_report(scan, [path])
    assert result['quantities']['dlp']['value'] is None
    assert result['unmatched_rdsr_events'] == 1


def test_invalid_units_rejected(tmp_path):
    path = rdsr(tmp_path / 'sr.dcm', generate_uid(), generate_uid(), unit='Gy')
    with pytest.raises(ValueError, match='unit'): read_rdsr([path])


def test_override_needs_source(dicom_series):
    folder, _ = dicom_series()
    with pytest.raises(ValueError, match='source'): dose_report(load_scan(folder), overrides={'ctdivol': 5})


def test_ssde_changes_with_output_and_patient_size(dicom_series):
    folder, _ = dicom_series(shape=(220, 240))
    scan = load_scan(folder)
    first = dose_report(scan, overrides={'ctdivol': 10, 'phantom_cm': 32, 'source': 'synthetic fixture'})
    second = dose_report(scan, overrides={'ctdivol': 20, 'phantom_cm': 32, 'source': 'synthetic fixture'})
    value = first['quantities']['ssde']['value']
    assert value > 0
    assert second['quantities']['ssde']['value'] == pytest.approx(2 * value)
    folder2, _ = dicom_series(shape=(180, 200))
    small = dose_report(load_scan(folder2), overrides={'ctdivol': 10, 'phantom_cm': 32, 'source': 'synthetic fixture'})
    assert small['quantities']['ssde']['value'] > value
    assert first == dose_report(scan, overrides={'ctdivol': 10, 'phantom_cm': 32, 'source': 'synthetic fixture'})


def test_distinct_acquisitions_sum_once_and_reconstruction_is_not_an_event(dicom_series, tmp_path):
    folder, _ = dicom_series(); scan = load_scan(folder); ds = scan.datasets[0]
    first = rdsr(tmp_path / 'first.dcm', ds.StudyInstanceUID, ds.IrradiationEventUID, dlp=150)
    second = rdsr(tmp_path / 'second.dcm', ds.StudyInstanceUID, generate_uid(), dlp=200)
    result = dose_report(scan, [first, second, first])
    assert result['quantities']['dlp']['value'] == 150
    assert result['provided_events_dlp_sum']['value'] == 350
    assert len(result['irradiation_events']) == 2


def test_partial_headers_do_not_qualify_ssde(dicom_series):
    folder, _ = dicom_series(shape=(220, 240)); scan = load_scan(folder)
    del scan.datasets[0].CTDIvol
    result = dose_report(scan, overrides={'phantom_cm': 32, 'source': 'fixture'})
    assert not result['quantities']['ctdivol']['complete']
    assert result['quantities']['ssde']['value'] is None


def test_kvp_override_requires_provenance(dicom_series):
    from ct_dose.imaging import load_scan
    from ct_dose.dosimetry import dose_report
    folder, _ = dicom_series()
    scan = load_scan(folder)
    import pytest
    with pytest.raises(ValueError, match='source'):
        dose_report(scan, overrides={'kvp': 120})
    result = dose_report(scan, overrides={'kvp': 120, 'source': 'Synthetic protocol worksheet for software test'})
    assert result['quantities']['kvp']['value'] == 120
    assert result['quantities']['kvp']['complete']
    assert result['quantities']['kvp']['source'].startswith('user:')

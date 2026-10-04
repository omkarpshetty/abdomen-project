import json
import pytest
import numpy as np
import SimpleITK as sitk
from ct_dose.cli import main
from ct_dose.imaging import load_scan, write_mask
from ct_dose.evaluation import segmentation_metrics


def test_cli_imported_masks_exports_and_separates_exploration(dicom_series, tmp_path):
    folder, _ = dicom_series(offsets=(0, 2, 4, 6, 8))
    scan = load_scan(folder)
    masks = tmp_path / 'masks'; masks.mkdir()
    mask = np.zeros_like(scan.array, bool); mask[1:4, 6:10, 6:10] = True
    write_mask(mask, scan.image, masks / 'liver.nii.gz')
    output = tmp_path / 'result'
    args = ['predict', str(folder), '--output', str(output), '--masks', str(masks), '--organs', 'liver', '--exploratory']
    assert main(args) == 0
    result = json.loads((output / 'report.json').read_text())
    assert result['organ_dose']['status'] == 'unavailable'
    assert result['measurements'][0]['volume_cm3'] == pytest.approx(0.144)
    assert len(list((output / 'annotated_slices').glob('*.png'))) == 5
    assert (output / 'prediction_features.csv').exists()
    assert 'estimate_mGy' not in (output / 'report.json').read_text()
    assert json.loads((output / 'exploratory.json').read_text())['values']
    assert main(args) == 2  # Never reuse stale outputs.


def test_reference_metrics_have_physical_units():
    reference = np.zeros((10, 10, 10), bool); reference[3:6, 3:6, 3:6] = True
    assert segmentation_metrics(reference, reference, (2, 1, 1)) == {'dice': 1.0, 'hd95_mm': 0.0, 'status': 'evaluated'}
    shifted = np.roll(reference, 1, axis=0)
    result = segmentation_metrics(shifted, reference, (2, 1, 1))
    assert result['dice'] < 1 and result['hd95_mm'] == 2


def test_failed_masks_preserve_metadata_report(dicom_series, tmp_path):
    folder, _ = dicom_series(); masks = tmp_path / 'empty'; masks.mkdir()
    output = tmp_path / 'result'
    assert main(['annotate', str(folder), '--output', str(output), '--masks', str(masks), '--organs', 'liver']) == 2
    result = json.loads((output / 'report.json').read_text())
    assert result['status'] == 'annotation_failed'
    assert result['dose']['quantities']['ctdivol']['value'] == 10


def test_correction_replaces_model_mask(dicom_series, tmp_path):
    folder, _ = dicom_series(offsets=(0, 2, 4, 6, 8)); scan = load_scan(folder)
    masks = tmp_path / 'masks'; masks.mkdir(); corrections = tmp_path / 'corrections'; corrections.mkdir()
    wrong = np.zeros_like(scan.array, bool); wrong[2, 5, 5] = True
    correct = np.zeros_like(scan.array, bool); correct[2, 5:8, 5:8] = True
    write_mask(wrong, scan.image, masks / 'liver.nii.gz')
    write_mask(correct, scan.image, corrections / 'liver.nii.gz')
    output = tmp_path / 'result'
    assert main(['annotate', str(folder), '--output', str(output), '--masks', str(masks), '--corrections', str(corrections), '--organs', 'liver']) == 0
    result = json.loads((output / 'report.json').read_text())
    assert result['organs']['liver']['source'] == 'expert_correction'
    assert result['measurements'][0]['volume_cm3'] == pytest.approx(9 * .003)


def test_pediatric_input_rejected(dicom_series, tmp_path):
    import pydicom
    folder, _ = dicom_series()
    for path in folder.glob('*.dcm'):
        ds = pydicom.dcmread(path); ds.PatientAge = '005Y'; ds.save_as(path)
    assert main(['annotate', str(folder), '--output', str(tmp_path / 'result'), '--adult-confirmed']) == 2


def test_ctorg_bilateral_evaluation(tmp_path):
    from scripts.prepare_ctorg import convert
    from ct_dose.evaluation import evaluate_masks
    labels = np.zeros((12, 12, 12), np.uint8)
    labels[3:5, 3:5, 3:5] = 4
    labels[3:5, 3:5, 8:10] = 4
    image = sitk.GetImageFromArray(labels)
    path = tmp_path / 'labels.nii.gz'; sitk.WriteImage(image, str(path))
    convert(path, tmp_path / 'reference')
    ct = tmp_path / 'ct.nii.gz'; sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), str(ct))
    masks = tmp_path / 'predicted'; masks.mkdir()
    left = labels == 4; left[:, :, 6:] = False
    right = labels == 4; right[:, :, :6] = False
    write_mask(left, image, masks / 'kidney_left.nii.gz')
    write_mask(right, image, masks / 'kidney_right.nii.gz')
    result = evaluate_masks(load_scan(ct), masks, tmp_path / 'reference', ['kidneys'])
    assert result['kidneys']['dice'] == 1


def test_legacy_adapters_use_unified_cli(monkeypatch):
    from ct_dose import compat
    calls = []
    monkeypatch.setattr(compat, 'main', lambda args: calls.append(args) or 0)
    assert compat.legacy('annotate', ['--dicom-dir'], ['--out'], ['--dicom-dir', 'scan', '--out', 'result', '--fast']) == 0
    assert calls == [['annotate', 'scan', '--output', 'result', '--fast']]
    with pytest.raises(SystemExit) as exc:
        compat.legacy('predict', ['--patient'], ['--output'], ['--patient'])
    assert exc.value.code == 2


def test_png_aspect_ratio_and_full_legend(tmp_path):
    from PIL import Image
    from src.annotate import save_annotated_series
    from color_map import ORGAN_COLORS
    volume = np.zeros((1, 20, 20))
    masks = {name: np.ones_like(volume, dtype=bool) for name in ORGAN_COLORS}
    save_annotated_series(volume, masks, str(tmp_path), pixel_spacing_xy=(2, 1))
    image = Image.open(tmp_path / 'annotated_0000.png')
    assert image.width == 512 + 260
    assert image.height >= 20 + 24 * len(ORGAN_COLORS)


def test_equivalent_physical_images_render_identically(tmp_path):
    from PIL import Image
    from ct_dose.segmentation import annotate
    values = np.zeros((3, 8, 8), dtype=np.float32)
    mask = np.zeros_like(values, bool); mask[1, 2:5, 1:3] = True
    views = []
    for index, flipped in enumerate([False, True]):
        folder = tmp_path / str(index); folder.mkdir()
        image = sitk.GetImageFromArray(values)
        if flipped:
            image.SetOrigin((7., 0., 0.)); image.SetDirection((-1., 0., 0., 0., 1., 0., 0., 0., 1.))
        ct = folder / 'ct.nii.gz'; sitk.WriteImage(image, str(ct))
        masks = folder / 'masks'; masks.mkdir()
        write_mask(mask[:, :, ::-1] if flipped else mask, image, masks / 'liver.nii.gz')
        annotate(load_scan(ct), folder / 'run', organs=['liver'], masks_dir=masks)
        views.append(np.asarray(Image.open(folder / 'run/annotated_slices/annotated_0001.png')))
    np.testing.assert_array_equal(*views)


def test_experimental_model_cannot_enter_primary_dose_report(dicom_series, tmp_path, monkeypatch):
    from ct_dose import modeling
    folder, _ = dicom_series(offsets=(0, 2, 4, 6, 8))
    scan = load_scan(folder)
    masks = tmp_path / 'masks'; masks.mkdir()
    mask = np.zeros_like(scan.array, bool); mask[2, 5:8, 5:8] = True
    write_mask(mask, scan.image, masks / 'liver.nii.gz')
    def experimental(frame, path, context):
        assert context['allow_experimental'] is True
        return np.array([3.]), {'model_sha256': 'test', 'dataset': {'dataset_id': 'SYNTHETIC', 'experimental_only': True}}
    monkeypatch.setattr(modeling, 'predict', experimental)
    output = tmp_path / 'result'
    assert main(['predict', str(folder), '--output', str(output), '--masks', str(masks),
                 '--organs', 'liver', '--model', 'fixture', '--experimental-model']) == 0
    assert json.loads((output / 'report.json').read_text())['organ_dose']['status'] == 'unavailable'
    assert not (output / 'organ_doses.csv').exists()
    assert (output / 'experimental_organ_doses.csv').exists()
    assert json.loads((output / 'experimental_model.json').read_text())['status'] == 'experimental_model_prediction'


def test_mask_statistics_match_integer_and_float_loader_paths(tmp_path):
    from ct_dose.imaging import Scan
    from ct_dose.segmentation import annotate
    rng = np.random.default_rng(42)
    pixels = rng.integers(-1000, 1000, size=(5, 20, 20), dtype=np.int16)
    mask = np.zeros_like(pixels, bool); mask[1:4, 2:18, 2:18] = True
    results = []
    for dtype in (np.int16, np.float32):
        image = sitk.GetImageFromArray(pixels.astype(dtype))
        masks = tmp_path / dtype.__name__; masks.mkdir()
        write_mask(mask, image, masks / 'liver.nii.gz')
        result = annotate(Scan(image, [], 'synthetic', None), masks / 'run', ['liver'], masks_dir=masks)
        results.append(result['measurements'])
    assert results[0] == results[1]

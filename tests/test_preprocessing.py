# tests/test_preprocessing.py
import os
import json
from src.preprocessing import preprocessing

def test_preprocess_fixture(tmp_path):
    fixture = os.path.join("tests", "fixtures", "sample_small.csv")
    assert os.path.exists(fixture), "Fixture missing: tests/fixtures/sample_small.csv"
    out = preprocessing.preprocess(fixture, config={'output_dir': str(tmp_path)})
    assert 'processed_path' in out
    assert os.path.exists(out['processed_path'])
    assert os.path.exists(out['validation_report_path'])
    assert os.path.exists(out['summary_metrics_path'])
    with open(out['validation_report_path'], 'r', encoding='utf-8') as f:
        vr = json.load(f)
    assert 'input_hash' in vr

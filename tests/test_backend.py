"""
IndicDoc AI - Backend API tests.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _encode(image: np.ndarray, ext: str = '.png') -> bytes:
    ok, buf = cv2.imencode(ext, image)
    assert ok
    return buf.tobytes()


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Test client isolated from the real outputs/ folder: no checkpoints, empty metrics dir."""
    from fastapi.testclient import TestClient
    import backend.main as main

    monkeypatch.setattr(main, 'METRICS_DIR', tmp_path)
    monkeypatch.setattr(main, 'CHECKPOINTS', {
        'doclayout': tmp_path / 'missing_doclayout.pt',
        'fasterrcnn': tmp_path / 'missing.pth',
        'yolov8': tmp_path / 'missing.pt',
    })
    main.get_model.cache_clear()
    return TestClient(main.app), main, tmp_path


class TestBackend:
    def test_root(self, api):
        client, _, _ = api
        assert client.get('/').status_code == 200

    def test_analyze_without_checkpoint_returns_labelled_demo(self, api):
        client, main, _ = api
        page = np.full((300, 200, 3), 255, np.uint8)
        r = client.post('/analyze', files={'file': ('page.png', _encode(page), 'image/png')})
        assert r.status_code == 200
        res = r.json()['results']
        assert res['is_demo'] is True
        assert (res['image_width'], res['image_height']) == (200, 300)
        assert len(res['boxes']) == len(res['labels']) == len(res['scores']) == len(res['class_names']) > 0
        for (x1, y1, x2, y2), label, name in zip(res['boxes'], res['labels'], res['class_names']):
            assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 300
            assert name == main.CLASS_NAMES[label]

    def test_analyze_rejects_non_image(self, api):
        client, _, _ = api
        r = client.post('/analyze', files={'file': ('notes.txt', b'not an image', 'text/plain')})
        assert r.status_code == 400
        assert r.json()['detail'] == 'Invalid image file'

    def test_analyze_rejects_unknown_model_type(self, api):
        client, _, _ = api
        page = np.full((50, 50, 3), 255, np.uint8)
        r = client.post('/analyze', params={'model_type': 'resnet'},
                        files={'file': ('p.png', _encode(page), 'image/png')})
        assert r.status_code == 422

    @pytest.mark.parametrize('conf', [-0.1, 1.5])
    def test_analyze_rejects_out_of_range_confidence(self, api, conf):
        client, _, _ = api
        page = np.full((50, 50, 3), 255, np.uint8)
        r = client.post('/analyze', params={'conf_threshold': conf},
                        files={'file': ('p.png', _encode(page), 'image/png')})
        assert r.status_code == 422

    def test_analyze_uses_checkpoint_when_present(self, api):
        import torch
        from ml.config import NUM_CLASSES
        from ml.models.baseline_frcnn import get_baseline_model

        client, main, tmp_path = api
        ckpt = tmp_path / 'baseline.pth'
        torch.save(get_baseline_model(NUM_CLASSES + 1, pretrained=False).state_dict(), ckpt)
        main.CHECKPOINTS['fasterrcnn'] = ckpt

        page = np.full((120, 80, 3), 255, np.uint8)
        r = client.post('/analyze', params={'model_type': 'fasterrcnn'},
                        files={'file': ('p.jpg', _encode(page, '.jpg'), 'image/jpeg')})
        assert r.status_code == 200
        assert r.json()['results']['is_demo'] is False

    def test_metrics_reads_saved_files(self, api):
        client, _, tmp_path = api
        (tmp_path / 'baseline_metrics.json').write_text(json.dumps({'precision': 0.12, 'map50': 0.3}))
        r = client.get('/metrics')
        assert r.status_code == 200
        assert r.json() == {'authors': None, 'baseline': {'precision': 0.12, 'map50': 0.3}, 'proposed': None}

    def test_analyze_rejects_oversized_upload(self, api, monkeypatch):
        client, main, _ = api
        monkeypatch.setattr(main, 'MAX_UPLOAD_BYTES', 1000)
        page = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)  # noisy PNG, well over 1000 bytes
        r = client.post('/analyze', files={'file': ('big.png', _encode(page), 'image/png')})
        assert r.status_code == 413

    def test_cors_only_allows_configured_origins(self, monkeypatch):
        import importlib
        from fastapi.testclient import TestClient
        monkeypatch.setenv('INDICDOC_CORS_ORIGINS', 'https://app.example.org')
        import backend.main as main
        main = importlib.reload(main)
        client = TestClient(main.app)
        ok = client.get('/', headers={'Origin': 'https://app.example.org'})
        bad = client.get('/', headers={'Origin': 'https://evil.example.com'})
        assert ok.headers.get('access-control-allow-origin') == 'https://app.example.org'
        assert 'access-control-allow-origin' not in bad.headers
        monkeypatch.delenv('INDICDOC_CORS_ORIGINS')
        importlib.reload(main)

    def test_low_confidence_threshold_reaches_the_model(self, api, monkeypatch):
        """YOLO drops boxes under its own 0.25 default unless the requested threshold is passed through."""
        client, main, tmp_path = api
        ckpt = tmp_path / 'proposed.pt'
        ckpt.write_bytes(b'placeholder')
        main.CHECKPOINTS['yolov8'] = ckpt
        seen = {}

        def fake_predict(model, image, model_type, device, conf=None):
            seen['conf'] = conf
            return {'boxes': np.array([[1., 1., 5., 5.]]), 'labels': np.array([0]), 'scores': np.array([0.12])}

        monkeypatch.setattr(main, 'get_model', lambda model_type: (object(), 'cpu'))
        monkeypatch.setattr(main, 'predict_single', fake_predict)
        page = np.full((40, 40, 3), 255, np.uint8)
        r = client.post('/analyze', params={'model_type': 'yolov8', 'conf_threshold': 0.1},
                        files={'file': ('p.png', _encode(page), 'image/png')})
        assert r.status_code == 200
        assert seen['conf'] == 0.1
        assert len(r.json()['results']['boxes']) == 1

    def test_default_threshold_is_the_models_recommended_one(self, api, monkeypatch):
        client, main, tmp_path = api
        (tmp_path / 'proposed_metrics.json').write_text(json.dumps({'map50': 0.3, 'recommended_conf': 0.3}))
        ckpt = tmp_path / 'proposed.pt'
        ckpt.write_bytes(b'placeholder')
        main.CHECKPOINTS['yolov8'] = ckpt
        seen = {}

        def fake_predict(model, image, model_type, device, conf=None):
            seen['conf'] = conf
            return {'boxes': np.zeros((0, 4)), 'labels': np.zeros(0), 'scores': np.zeros(0)}

        monkeypatch.setattr(main, 'get_model', lambda model_type: (object(), 'cpu'))
        monkeypatch.setattr(main, 'predict_single', fake_predict)
        page = np.full((40, 40, 3), 255, np.uint8)
        r = client.post('/analyze', params={'model_type': 'yolov8'}, files={'file': ('p.png', _encode(page), 'image/png')})
        assert r.status_code == 200
        assert seen['conf'] == 0.3 and r.json()['results']['conf_threshold'] == 0.3

    def test_rejects_huge_pixel_count(self, api):
        import io
        from PIL import Image
        client, _, _ = api
        buf = io.BytesIO()
        Image.new('1', (12000, 12000), 1).save(buf, format='PNG')
        r = client.post('/analyze', files={'file': ('bomb.png', buf.getvalue(), 'image/png')})
        assert r.status_code == 413

    def test_server_stays_responsive_during_inference(self, api, monkeypatch):
        """Inference runs in a worker thread, so a slow page does not block other requests."""
        import asyncio
        import time as _time
        import httpx
        client, main, tmp_path = api
        ckpt = tmp_path / 'proposed.pt'
        ckpt.write_bytes(b'placeholder')
        main.CHECKPOINTS['yolov8'] = ckpt

        def slow_predict(model, image, model_type, device, conf=None):
            _time.sleep(1.0)
            return {'boxes': np.zeros((0, 4)), 'labels': np.zeros(0), 'scores': np.zeros(0)}

        monkeypatch.setattr(main, 'get_model', lambda model_type: (object(), 'cpu'))
        monkeypatch.setattr(main, 'predict_single', slow_predict)
        page = _encode(np.full((40, 40, 3), 255, np.uint8))

        async def run():
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as ac:
                start = _time.perf_counter()
                analyze = asyncio.create_task(ac.post('/analyze', params={'model_type': 'yolov8'},
                                                      files={'file': ('p.png', page, 'image/png')}))
                await asyncio.sleep(0.1)
                root = await ac.get('/')
                root_time = _time.perf_counter() - start
                await analyze
                return root.status_code, root_time

        status, root_time = asyncio.run(run())
        assert status == 200 and root_time < 0.6

    def test_bad_upload_never_triggers_a_runtime_pip_install(self, api, monkeypatch):
        """Ultralytics patches PIL.Image.open to pip-install pi-heif when an image fails to open; serving must not."""
        import subprocess
        import ultralytics  # noqa: F401  (activates the PIL patch)
        client, _, _ = api
        calls = []
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 1))
        monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: calls.append(a) or b'')
        r = client.post('/analyze', files={'file': ('empty.png', b'', 'image/png')})
        r2 = client.post('/analyze', files={'file': ('junk.png', b'\x89PNG garbage', 'image/png')})
        assert r.status_code == 400 and r2.status_code == 400
        assert not [c for c in calls if 'pip' in str(c)]

    def test_models_are_loaded_at_startup(self, api, monkeypatch):
        """The first visitor should not wait for a model to load: available checkpoints load when the API starts."""
        from fastapi.testclient import TestClient
        _, main, tmp_path = api
        ckpt = tmp_path / 'proposed.pt'
        ckpt.write_bytes(b'placeholder')
        main.CHECKPOINTS['yolov8'] = ckpt          # baseline checkpoint stays missing
        loaded = []
        monkeypatch.setattr(main, 'get_model', lambda model_type: loaded.append(model_type) or (object(), 'cpu'))
        with TestClient(main.app) as client:      # entering the context runs the startup hook
            assert client.get('/').status_code == 200
        assert loaded == ['yolov8']

    @staticmethod
    def _fake_model(main, tmp_path, monkeypatch, model_type='doclayout'):
        ckpt = tmp_path / f'{model_type}.pt'
        ckpt.write_bytes(b'placeholder')
        main.CHECKPOINTS[model_type] = ckpt
        monkeypatch.setattr(main, 'get_model', lambda mt: (object(), 'cpu'))
        monkeypatch.setattr(main, 'predict_single', lambda *a, **k: {
            'boxes': np.array([[1., 1., 30., 10.], [1., 15., 39., 39.]]), 'labels': np.array([26, 21]),
            'scores': np.array([0.9, 0.8])})

    def test_default_model_is_the_authors_doclayout(self, api, monkeypatch):
        client, main, tmp_path = api
        self._fake_model(main, tmp_path, monkeypatch)
        r = client.post('/analyze', files={'file': ('p.png', _encode(np.full((40, 40, 3), 255, np.uint8)), 'image/png')})
        assert r.status_code == 200
        assert r.json()['results']['model_type'] == 'doclayout'
        assert r.json()['results']['class_names'] == ['section-title', 'paragraph']

    def test_ocr_returns_structured_document(self, api, monkeypatch):
        client, main, tmp_path = api
        self._fake_model(main, tmp_path, monkeypatch)
        seen = {}

        def fake_extract(image, detections, languages='auto'):
            seen['languages'] = languages
            return {'script': None, 'languages': languages, 'markdown': '## Title\n\nBody\n',
                    'regions': [{'label': 'section-title', 'box': [1, 1, 30, 10], 'score': 0.9, 'text': 'Title'}],
                    'words_on_page': 2, 'words_in_regions': 2}

        monkeypatch.setattr(main, 'extract_document', fake_extract)
        monkeypatch.setattr(main, 'tesseract_available', lambda: True)
        page = _encode(np.full((40, 40, 3), 255, np.uint8))
        r = client.post('/analyze', params={'ocr': 'true', 'ocr_lang': 'hin+eng'}, files={'file': ('p.png', page, 'image/png')})
        assert r.status_code == 200
        doc = r.json()['results']['document']
        assert doc['markdown'].startswith('## Title') and doc['regions'][0]['text'] == 'Title'
        assert seen['languages'] == 'hin+eng'

    @pytest.mark.parametrize('lang', ['xyz', 'hin;rm -rf /', 'HIN'])
    def test_ocr_rejects_unknown_languages(self, api, monkeypatch, lang):
        client, main, tmp_path = api
        self._fake_model(main, tmp_path, monkeypatch)
        monkeypatch.setattr(main, 'tesseract_available', lambda: True)
        page = _encode(np.full((40, 40, 3), 255, np.uint8))
        r = client.post('/analyze', params={'ocr': 'true', 'ocr_lang': lang}, files={'file': ('p.png', page, 'image/png')})
        assert r.status_code in (400, 422)

    def test_ocr_unavailable_gives_503(self, api, monkeypatch):
        client, main, tmp_path = api
        self._fake_model(main, tmp_path, monkeypatch)
        monkeypatch.setattr(main, 'tesseract_available', lambda: False)
        page = _encode(np.full((40, 40, 3), 255, np.uint8))
        r = client.post('/analyze', params={'ocr': 'true'}, files={'file': ('p.png', page, 'image/png')})
        assert r.status_code == 503

    def test_mode_all_uses_high_recall_threshold_and_dedupes(self, api, monkeypatch):
        client, main, tmp_path = api
        (tmp_path / 'authors_metrics.json').write_text(json.dumps({'recommended_conf': 0.35, 'high_recall_conf': 0.15}))
        ckpt = tmp_path / 'doclayout.pt'
        ckpt.write_bytes(b'placeholder')
        main.CHECKPOINTS['doclayout'] = ckpt
        monkeypatch.setattr(main, 'get_model', lambda mt: (object(), 'cpu'))
        monkeypatch.setattr(main, 'predict_single', lambda *a, **k: {      # one heading under three labels + a paragraph
            'boxes': np.array([[0., 0., 30., 8.], [0., 0., 30., 8.], [0., 0., 31., 8.], [0., 12., 39., 39.]]),
            'labels': np.array([26, 30, 32, 21]), 'scores': np.array([0.33, 0.27, 0.2, 0.8])})
        page = _encode(np.full((40, 40, 3), 255, np.uint8))
        r = client.post('/analyze', params={'mode': 'all'}, files={'file': ('p.png', page, 'image/png')}).json()['results']
        assert r['conf_threshold'] == 0.15 and r['mode'] == 'all'
        assert sorted(r['class_names']) == ['paragraph', 'section-title']
        r = client.post('/analyze', files={'file': ('p.png', page, 'image/png')}).json()['results']
        assert r['conf_threshold'] == 0.35 and r['class_names'] == ['paragraph']


class TestVerifyEndpoint:
    def test_profiles_listed(self, api):
        client, _, _ = api
        profiles = client.get('/verify/profiles').json()
        assert {'question_paper', 'office_memorandum'} <= set(profiles)

    def test_needs_model_and_ocr(self, api):
        client, _, _ = api
        r = client.post('/verify', files={'file': ('p.png', _encode(np.full((50, 50, 3), 255, np.uint8)), 'image/png')})
        assert r.status_code == 503

    def test_rejects_bad_parameters(self, api, monkeypatch):
        client, main, tmp_path = api
        ckpt = tmp_path / 'doclayout.pt'
        ckpt.write_bytes(b'x')
        monkeypatch.setitem(main.CHECKPOINTS, 'doclayout', ckpt)
        monkeypatch.setattr(main, 'tesseract_available', lambda: True)
        page = ('p.png', _encode(np.full((50, 50, 3), 255, np.uint8)), 'image/png')
        assert client.post('/verify?profile=nope', files={'file': page}).status_code == 400
        assert client.post('/verify?bilingual=Latin', files={'file': page}).status_code == 400
        assert client.post('/verify?lang=xyz', files={'file': page}).status_code == 400
        bad_bank = ('b.json', b'{"not": "a list"}', 'application/json')
        assert client.post('/verify', files={'file': page, 'bank': bad_bank}).status_code == 400


class TestVerifyConsensus:
    """POST /verify with both readers (their page readings stubbed): the response carries the consensus."""

    def _setup(self, api, monkeypatch):
        from ml.verify.reader import read_pages
        from tests.test_consensus import EN_PAGE, _ocr_page_misreading_q2
        client, main, tmp_path = api
        ckpt = tmp_path / 'doclayout.pt'
        ckpt.write_bytes(b'x')
        monkeypatch.setitem(main.CHECKPOINTS, 'doclayout', ckpt)
        monkeypatch.setattr(main, 'tesseract_available', lambda: True)
        monkeypatch.setattr(main, '_read_paper', lambda images, lang: _ocr_page_misreading_q2())
        monkeypatch.setattr(main, 'second_reading',
                            lambda images, pages, pdf, dpi: read_pages([EN_PAGE], source='text-layer'))
        monkeypatch.setattr(main, 'make_rereader', lambda images: None)
        return client

    def test_json_has_consensus(self, api, monkeypatch):
        client = self._setup(api, monkeypatch)
        page = ('p.png', _encode(np.full((60, 60, 3), 255, np.uint8)), 'image/png')
        rep = client.post('/verify', files={'file': page}).json()
        assert 'consensus' in rep and rep['consensus']['readers']['b'] == 'layout-free, PDF text layer'
        assert not [i for i in rep['issues'] if i['code'] == 'NUMBER_GAP']        # reader A's misreading
        assert any(d['code'] == 'NUMBER_GAP' for d in rep['consensus']['dismissed'])

    def test_single_reader_on_request(self, api, monkeypatch):
        client = self._setup(api, monkeypatch)
        page = ('p.png', _encode(np.full((60, 60, 3), 255, np.uint8)), 'image/png')
        rep = client.post('/verify?second_reader=false', files={'file': page}).json()
        assert 'consensus' not in rep
        assert any(i['code'] == 'NUMBER_GAP' for i in rep['issues'])

    def test_html_report(self, api, monkeypatch):
        client = self._setup(api, monkeypatch)
        page = ('p.png', _encode(np.full((60, 60, 3), 255, np.uint8)), 'image/png')
        r = client.post('/verify?output=html', files={'file': page})
        assert r.status_code == 200 and 'Dual-reader consensus' in r.text

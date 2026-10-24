"""
IndicDoc AI - Streamlit app tests (headless, via streamlit.testing).
"""
from pathlib import Path

import pytest

APP = str(Path(__file__).parent.parent / 'app' / 'streamlit_app.py')


@pytest.fixture
def no_models(tmp_path, monkeypatch):
    """Run the app as if no trained checkpoints exist (demo mode), whatever is in outputs/models."""
    import ml.config as config
    monkeypatch.setattr(config, 'BASELINE_CHECKPOINT', tmp_path / 'missing.pth')
    monkeypatch.setattr(config, 'PROPOSED_CHECKPOINT', tmp_path / 'missing.pt')
    monkeypatch.setattr(config, 'AUTHORS_CHECKPOINT', tmp_path / 'missing_authors.pt')


class TestStreamlitApp:
    def test_app_renders_all_tabs(self, no_models):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=120).run()
        assert not at.exception
        assert [t.label for t in at.tabs] == ['Verify Paper', 'Analyze', 'Dashboard', 'Model Insights', 'Evaluation', 'About']

    def test_results_list_appears_right_after_analyze(self, no_models):
        """The detected-elements list must show on the same click, not one interaction later."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.checkbox[0].check().run()  # use a bundled sample image
        [b for b in at.button if b.label == 'Analyze Document'][0].click().run()
        assert not at.exception
        n_detected = len(at.session_state['last_results']['boxes'])
        listed = [m.value for m in at.markdown if m.value.startswith('`')]
        assert n_detected > 0 and len(listed) > 0

    def test_shows_saved_results(self, tmp_path, monkeypatch, no_models):
        """With evaluation outputs present, the app shows them: formatted numbers, per-language rows, loss curve."""
        import json
        import ml.config as config
        from streamlit.testing.v1 import AppTest

        metrics_dir = tmp_path / 'metrics'
        metrics_dir.mkdir()
        fake = {'model': 'Faster R-CNN (ResNet-50 FPN)', 'precision': 0.123456, 'recall': 0.5, 'f1': 0.2, 'map50': 0.31,
                'map50_95': 0.2, 'avg_iou': 0.7, 'inference_time': 0.25, 'model_size_mb': 165.0,
                'total_predictions': 10, 'num_images': 40, 'dataset': 'IndicDLP test subset', 'note': 'test data',
                'per_group': {'hindi': {'images': 30, 'map50': 0.4, 'map50_95': 0.25},
                              'tamil': {'images': 10, 'map50': 0.2, 'map50_95': 0.1}}}
        (metrics_dir / 'baseline_metrics.json').write_text(json.dumps(fake))
        (metrics_dir / 'baseline_training_history.json').write_text(json.dumps({'train_losses': [2.0, 1.5], 'val_losses': [1.9, 1.6]}))
        monkeypatch.setattr(config, 'METRICS_DIR', metrics_dir)
        monkeypatch.setattr(config, 'PLOTS_DIR', tmp_path / 'plots')
        monkeypatch.setattr(config, 'PREDICTION_DIR', tmp_path / 'predictions')

        at = AppTest.from_file(APP, default_timeout=120).run()
        assert not at.exception
        html = ' '.join(m.value for m in at.markdown)
        assert '0.123' in html and '0.123456' not in html
        assert '<td>Hindi</td><td>30</td><td>0.400</td>' in html
        assert 'IndicDLP test subset (40 pages)' in html
        assert len(at.get('vega_lite_chart')) >= 1  # st.line_chart renders as a Vega-Lite chart

    def test_model_choices_follow_available_checkpoints(self, tmp_path, monkeypatch):
        """Only models with a checkpoint are offered; demo mode only when there is none."""
        import torch
        import ml.config as config
        from ml.models.baseline_frcnn import get_baseline_model
        from streamlit.testing.v1 import AppTest

        ckpt = tmp_path / 'baseline.pth'
        torch.save(get_baseline_model(config.NUM_CLASSES + 1, pretrained=False).state_dict(), ckpt)
        monkeypatch.setattr(config, 'BASELINE_CHECKPOINT', ckpt)
        monkeypatch.setattr(config, 'PROPOSED_CHECKPOINT', tmp_path / 'missing.pt')
        monkeypatch.setattr(config, 'AUTHORS_CHECKPOINT', tmp_path / 'missing_authors.pt')

        at = AppTest.from_file(APP, default_timeout=180).run()
        assert at.sidebar.selectbox[0].options == ['Faster R-CNN (Baseline)']
        at.checkbox[0].check().run()
        [b for b in at.button if b.label == 'Analyze Document'][0].click().run()
        assert not at.exception and not at.warning
        assert at.session_state['last_results'].get('is_demo', False) is False

    def test_confidence_slider_starts_at_recommended_threshold(self, tmp_path, monkeypatch):
        import json
        import torch
        import ml.config as config
        from ml.models.baseline_frcnn import get_baseline_model
        from streamlit.testing.v1 import AppTest

        ckpt = tmp_path / 'baseline.pth'
        torch.save(get_baseline_model(config.NUM_CLASSES + 1, pretrained=False).state_dict(), ckpt)
        metrics_dir = tmp_path / 'metrics'
        metrics_dir.mkdir()
        (metrics_dir / 'baseline_metrics.json').write_text(json.dumps({'map50': 0.2, 'recommended_conf': 0.35}))
        monkeypatch.setattr(config, 'BASELINE_CHECKPOINT', ckpt)
        monkeypatch.setattr(config, 'PROPOSED_CHECKPOINT', tmp_path / 'missing.pt')
        monkeypatch.setattr(config, 'AUTHORS_CHECKPOINT', tmp_path / 'missing_authors.pt')
        monkeypatch.setattr(config, 'METRICS_DIR', metrics_dir)

        at = AppTest.from_file(APP, default_timeout=180).run()
        assert at.sidebar.slider[0].value == pytest.approx(0.35)

    def test_samples_are_real_indicdlp_pages(self, no_models):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.checkbox[0].check().run()
        options = [s for s in at.selectbox if any(o.endswith('.png') for o in s.options)][0].options
        assert 'hindi_newspapers.png' in options and not any(o.startswith('sample_') for o in options)

    def test_results_do_not_carry_over_to_another_page(self, no_models):
        """Boxes found on one page must not be drawn on the next page before it is analysed."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.checkbox[0].check().run()
        sample_box = [s for s in at.selectbox if 'hindi_newspapers.png' in s.options][0]
        [b for b in at.button if b.label == 'Analyze Document'][0].click().run()
        assert [m.value for m in at.markdown if m.value.startswith('`')]   # results shown for page 1
        sample_box = [s for s in at.selectbox if 'hindi_newspapers.png' in s.options][0]
        other = [o for o in sample_box.options if o != sample_box.value][0]
        sample_box.select(other).run()
        assert not [m.value for m in at.markdown if m.value.startswith('`')]   # nothing until analysed

    def test_reset_shows_the_page_again(self, no_models):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.checkbox[0].check().run()
        [b for b in at.button if b.label == 'Analyze Document'][0].click().run()
        [b for b in at.button if b.label == 'Reset'][0].click().run()
        assert not at.exception
        assert [m.value for m in at.markdown if m.value.startswith('`')] == []
        captions = [img.caption for el in at.get('image') for img in el.proto.imgs]
        assert 'Original Document' in captions   # the page itself is still displayed

    def test_authors_model_is_default_when_present(self, tmp_path, monkeypatch):
        import ml.config as config
        from streamlit.testing.v1 import AppTest
        for name in ('AUTHORS_CHECKPOINT', 'PROPOSED_CHECKPOINT', 'BASELINE_CHECKPOINT'):
            f = tmp_path / f'{name}.pt'
            f.write_bytes(b'x')
            monkeypatch.setattr(config, name, f)
        at = AppTest.from_file(APP, default_timeout=120).run()
        assert at.sidebar.selectbox[0].options[0] == 'DocLayout-YOLO (IndicDLP authors)'
        assert len(at.sidebar.selectbox[0].options) == 3

    def test_slider_filters_results_without_reanalysing(self, no_models):
        """Lowering the threshold after analysis must show more boxes immediately."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.checkbox[0].check().run()
        [b for b in at.button if b.label == 'Analyze Document'][0].click().run()
        scores = at.session_state['last_results']['scores']
        at.sidebar.slider[0].set_value(0.95).run()
        high = len([m for m in at.markdown if m.value.startswith('`')])
        at.sidebar.slider[0].set_value(0.05).run()
        low = len([m for m in at.markdown if m.value.startswith('`')])
        assert high == sum(s >= 0.95 for s in scores) and low == len(scores) and low > high

    def test_ocr_reads_text_of_a_real_sample_page(self, no_models):
        import shutil
        from ml.ocr import tesseract_available
        from streamlit.testing.v1 import AppTest
        if not tesseract_available():
            pytest.skip('tesseract or OCR models missing')
        at = AppTest.from_file(APP, default_timeout=300).run()
        at.checkbox[0].check().run()
        sample = [s for s in at.selectbox if 'english_research_papers.png' in s.options][0]
        sample.select('english_research_papers.png').run()
        [b for b in at.button if b.label == 'Analyze Document'][0].click().run()
        [b for b in at.button if b.label == 'Read text (OCR)'][0].click().run()
        assert not at.exception
        doc = at.session_state['ocr_result']
        assert doc['words_on_page'] > 100 and doc['languages'] == 'eng'
        assert any('Machine-read text' in c.value for c in at.caption)

    def test_detection_mode_sets_the_threshold(self, tmp_path, monkeypatch):
        import json
        import ml.config as config
        from streamlit.testing.v1 import AppTest
        ckpt = tmp_path / 'authors.pt'
        ckpt.write_bytes(b'x')
        metrics_dir = tmp_path / 'metrics'
        metrics_dir.mkdir()
        (metrics_dir / 'authors_metrics.json').write_text(json.dumps({'map50': 0.7, 'recommended_conf': 0.35, 'high_recall_conf': 0.15}))
        monkeypatch.setattr(config, 'AUTHORS_CHECKPOINT', ckpt)
        monkeypatch.setattr(config, 'PROPOSED_CHECKPOINT', tmp_path / 'missing.pt')
        monkeypatch.setattr(config, 'BASELINE_CHECKPOINT', tmp_path / 'missing.pth')
        monkeypatch.setattr(config, 'METRICS_DIR', metrics_dir)
        at = AppTest.from_file(APP, default_timeout=120).run()
        assert at.sidebar.radio[0].value == 'Find every part'
        assert at.sidebar.slider[0].value == pytest.approx(0.15)
        at.sidebar.radio[0].set_value('Balanced').run()
        assert at.sidebar.slider[0].value == pytest.approx(0.35)

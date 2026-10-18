"""
IndicDoc AI - Tests for the IndicDLP subset converter (pure functions, no network).
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))


def _png(w, h):
    buf = io.BytesIO()
    Image.new('RGB', (w, h), 'white').save(buf, format='PNG')
    return buf.getvalue()


class TestParseFilename:
    @pytest.mark.parametrize('name,expected', [
        ('ar_as_001025_0.png', ('acts & rules', 'assamese')),
        ('rp_ml_000322_0.png', ('research papers', 'malayalam')),
        ('np_en_000001_3.png', ('newspapers', 'english')),
    ])
    def test_known_codes(self, name, expected):
        from prepare_indicdlp import parse_filename
        assert parse_filename(name) == expected

    def test_unknown_codes_are_kept_not_dropped(self):
        from prepare_indicdlp import parse_filename
        assert parse_filename('zz_qq_1_0.png') == ('zz', 'qq')


class TestRowsToCoco:
    def test_converts_and_cleans_boxes(self, tmp_path):
        from prepare_indicdlp import rows_to_coco
        rows = [{'image': {'bytes': _png(100, 200), 'path': 'tb_hi_000001_0.png'},
                 'bboxes': [[10, 20, 30, 40],      # fine
                            [90, 190, 30, 30],     # overflows the page: clipped to 10x10
                            [5, 5, 0.5, 10],       # degenerate: dropped
                            [150, 10, 10, 10]],    # fully outside: dropped
                 'category_ids': [21, 36, 21, 14]}]
        coco, stats = rows_to_coco(rows, tmp_path)

        assert (tmp_path / 'tb_hi_000001_0.png').exists()
        img = coco['images'][0]
        assert (img['width'], img['height'], img['language'], img['domain']) == (100, 200, 'hindi', 'textbooks')
        anns = coco['annotations']
        assert [a['category_id'] for a in anns] == [21, 36]
        assert anns[0]['bbox'] == [10, 20, 30, 40] and anns[0]['area'] == 1200
        assert anns[1]['bbox'] == [90, 190, 10, 10]
        assert stats == {'images': 1, 'boxes_in': 4, 'boxes_kept': 2, 'boxes_clipped': 1, 'boxes_dropped': 2}
        assert len(coco['categories']) == 42 and coco['categories'][21]['name'] == 'paragraph'

    def test_rejects_unknown_class_id(self, tmp_path):
        from prepare_indicdlp import rows_to_coco
        rows = [{'image': {'bytes': _png(50, 50), 'path': 'tb_hi_1_0.png'}, 'bboxes': [[1, 1, 5, 5]], 'category_ids': [42]}]
        with pytest.raises(ValueError):
            rows_to_coco(rows, tmp_path)


class TestPlanSelection:
    def test_spreads_picks_across_shards_per_language(self):
        """Candidates are (shard, row group) pairs that contain the language; picks prefer different shards."""
        from prepare_indicdlp import plan_selection
        candidates = {'hindi': [('h0', 0), ('h0', 1), ('h1', 0), ('h2', 3), ('h3', 1)],
                      'tamil': [('t0', 0), ('t0', 1), ('t0', 2)]}
        plan = plan_selection(candidates, groups_per_language=2, seed=0)
        hindi = [p for p in plan if p[0] == 'hindi']
        assert len(hindi) == 2 and len({p[1] for p in hindi}) == 2   # two different shards
        tamil = [p for p in plan if p[0] == 'tamil']
        assert len(tamil) == 2 and len({p[2] for p in tamil}) == 2   # one shard, two different row groups
        assert plan == plan_selection(candidates, groups_per_language=2, seed=0)  # reproducible


class TestLanguageCandidates:
    def test_row_groups_straddling_two_languages_count_for_both(self):
        """Shards are sorted by language but can cross a boundary; each row group is labelled by its own pages."""
        from prepare_indicdlp import language_candidates
        row_group_paths = {
            ('s0', 0): ['tb_as_1_0.png'] * 80 + ['tb_bn_1_0.png'] * 20,   # Assamese -> Bengali boundary
            ('s0', 1): ['tb_bn_2_0.png'] * 100,
            ('s1', 0): ['tb_bn_3_0.png'] * 5 + ['tb_en_1_0.png'] * 95,
        }
        cands = language_candidates(row_group_paths, min_pages=10)
        assert cands['assamese'] == [('s0', 0)]
        assert cands['bengali'] == [('s0', 0), ('s0', 1)]   # 5 Bengali pages in s1:0 is below min_pages
        assert cands['english'] == [('s1', 0)]

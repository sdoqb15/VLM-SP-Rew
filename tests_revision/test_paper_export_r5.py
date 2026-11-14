"""Opt-in field-pipeline checks on synthetic provider/cache fixtures only."""
from pathlib import Path
import json
import shutil
import unittest
import test_offline_export as legacy
from annotation_cache_manifest import sha
from refine_room_types import cluster_rooms_by_similarity


class PaperExporterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = legacy.ExporterTests()
        self.fixture.setUp()
        self.fixture.config = Path(__file__).parents[1] / 'config_paper_r5.yaml'

    def tearDown(self):
        self.fixture.tearDown()

    def call(self, **kwargs):
        return self.fixture.call(paper_contract=True, **kwargs)

    def test_exact_paper_hops_and_values_with_source_bytes_unchanged(self):
        f = self.fixture
        before = {p.relative_to(f.root).as_posix(): sha(p.read_bytes()) for p in f.root.rglob('*') if p.is_file()}
        field, complete = self.call()
        after = {p.relative_to(f.root).as_posix(): sha(p.read_bytes()) for p in f.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(field['method_contract'], 'original_manuscript_r5')
        self.assertEqual([p['graph_hops'] for p in field['points']], [2., 1.])
        self.assertEqual([p['value'] for p in field['points']], [5., 5.])
        self.assertFalse(field['source_integrity_qualified'])
        self.assertTrue(field['synthetic'])
        self.assertTrue(field['paper_contract']['runtime_observation_reward_and_full_benchmark_qualification_pending'])
        self.assertTrue(complete['no_API_simulator_or_training_calls'])

    def test_original_infinite_distance_repair_covers_connected_unreachable_components(self):
        f = self.fixture
        shutil.copytree(f.fixture.point, f.root / 'pos_10.000_0.000')
        shutil.copytree(f.fixture.point, f.root / 'pos_11.000_0.000')
        field, _ = self.call()
        repaired = [p for p in field['points'] if p['position_repaired']]
        self.assertEqual(len(repaired), 2)
        self.assertTrue(all(p['graph_hops'] == 2 for p in repaired))
        paths = json.loads((f.out_parent / 'out/PATH_POSITIONS.json').read_text())
        self.assertTrue(all(r['added_one'] and r['graph_edges_not_created'] for r in paths['repairs']))
        self.assertTrue(all(paths['original_paths'][str(r['node_id'])] is None for r in paths['repairs']))

    def test_original_fixed_target_is_not_implicitly_replaced_by_all_instances(self):
        f = self.fixture
        f.meta['goal_instances'].append(dict(object_id='SECOND_ENGINEERING_PLANT', position=dict(x=0., y=0., z=0.)))
        f.metadata.write_text(json.dumps(f.meta))
        with self.assertRaisesRegex(ValueError, 'one explicit goal instance'):
            self.call()
        self.assertFalse((f.out_parent / 'out').exists())

    def test_cannot_accidentally_use_the_legacy_threshold_configuration(self):
        f = self.fixture
        f.config = Path(__file__).parents[1] / 'config.yaml'
        with self.assertRaisesRegex(ValueError, 'explicit paper-contract'):
            self.call()
        self.assertFalse((f.out_parent / 'out').exists())

    def test_paper_repair_and_legacy_repair_flags_cannot_be_conflated(self):
        with self.assertRaisesRegex(ValueError, 'do not mix'):
            self.call(repair_all_unreachable=True)

    def test_jaccard_only_and_spatial_mixture_are_not_assumed_equivalent(self):
        # Jaccard=2/5=.4 > .3, but .7*Jaccard + .3*exp(-100^2/18)
        # is about .28 < .3: this is an analytic partition counterexample.
        points = [dict(objects=['a','b','c','d','e'], position=(0,0)), dict(objects=['a','b'], position=(100,0))]
        paper = cluster_rooms_by_similarity(points, similarity_threshold=.3, distance_weight=0.)
        prototype = cluster_rooms_by_similarity(points, similarity_threshold=.3, distance_weight=.3)
        self.assertEqual(len(set(paper)), 1)
        self.assertEqual(len(set(prototype)), 2)


if __name__ == '__main__':
    unittest.main()

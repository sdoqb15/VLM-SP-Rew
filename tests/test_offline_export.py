"""Field-export unit tests using generated annotations, images and scene geometry."""
from pathlib import Path
import hashlib,json,shutil,tempfile,unittest
import test_annotation_cache as cachefixtures
from vlm_sp_rew.annotation_cache import sha
from vlm_sp_rew.export_field import export_cache

class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.fixture=cachefixtures.CacheTests();self.fixture.setUp()
        self.root=self.fixture.root
        self.other=tempfile.TemporaryDirectory();self.out_parent=Path(self.other.name)
        shutil.copytree(self.fixture.point,self.root/'pos_1.000_0.000')
        scene={'rooms':[{'floorPolygon':[{'x':0,'z':0},{'x':0,'z':12},
                                         {'x':12,'z':12},{'x':12,'z':0}]}]}
        self.scene=self.root/'ENGINEERING_SCENE.json';self.scene.write_text(json.dumps(scene))
        self.metadata=self.root/'ENGINEERING_GOAL_METADATA.json'
        self.meta=dict(synthetic=True,scene_sha256=sha(self.scene.read_bytes()),
                       goal_reference_sha256=sha(self.fixture.goal.read_bytes()),
                       goal_category='HousePlant',
                       goal_instances=[dict(object_id='ENGINEERING_PLANT',position={'x':2.,'y':0.,'z':0.})])
        self.metadata.write_text(json.dumps(self.meta))
        self.config=Path(__file__).parents[1]/'src/vlm_sp_rew/configs/paper.yaml'
    def tearDown(self):
        self.fixture.tearDown();self.other.cleanup()
    def call(self,name='out',**kwargs):
        return export_cache(self.root,self.scene,self.metadata,self.fixture.goal,
                            self.config,self.out_parent/name,**kwargs)
    def test_exports_paper_hops_and_value_bands_without_touching_source(self):
        before={p.relative_to(self.root).as_posix():sha(p.read_bytes())for p in self.root.rglob('*')if p.is_file()}
        field,complete=self.call()
        after={p.relative_to(self.root).as_posix():sha(p.read_bytes())for p in self.root.rglob('*')if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual([p['graph_path_position']for p in field['points']],[2.,1.])
        self.assertEqual([p['value']for p in field['points']],[5,5])
        self.assertTrue(field['synthetic']);self.assertFalse(field['source_integrity_qualified'])
        self.assertTrue(complete['no_API_simulator_or_training_calls'])
        exported={item['path']:item for item in complete['files']}
        self.assertTrue({'FIELD.json','GRAPH.json','PATH_POSITIONS.json'}<=set(exported))
        for name,item in exported.items():
            self.assertEqual(sha((self.out_parent/'out'/name).read_bytes()),item['sha256'])
    def test_never_overwrites_completed_output(self):
        self.call()
        with self.assertRaisesRegex(ValueError,'fresh output directory'):self.call()
    def test_scene_mismatch_fails_before_creating_output(self):
        self.meta['scene_sha256']='0'*64;self.metadata.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError,'another scene'):self.call()
        self.assertFalse((self.out_parent/'out').exists())
    def test_legacy_unqualified_cache_is_not_silently_accepted(self):
        for p in self.root.glob('pos_*/api_audit/*/COMPLETE.json'):
            p.rename(p.with_name('REMOVED_ENGINEERING_COMPLETE.json'))
        with self.assertRaisesRegex(ValueError,'No immutable provider provenance'):self.call()
        field,complete=self.call(allow_parsed_only=True)
        self.assertFalse(field['source_integrity_qualified'])
        self.assertFalse(field['annotation_record_integrity_qualified'])
    def test_paper_disconnected_repair_keeps_original_graph_paths(self):
        shutil.copytree(self.fixture.point,self.root/'pos_10.000_0.000')
        shutil.copytree(self.fixture.point,self.root/'pos_11.000_0.000')
        field,complete=self.call()
        repaired=[p for p in field['points']if p['position_repaired']]
        self.assertEqual(len(repaired),2)
        paths=json.loads((self.out_parent/'out/PATH_POSITIONS.json').read_text())
        self.assertTrue(all(paths['original_paths'][str(p['id'])] is None for p in repaired))
        self.assertTrue(all(x['inherited_hops'] > 0 for x in paths['repairs']))
    def test_missing_views_never_become_fabricated_annotations(self):
        (self.root/'pos_1.000_0.000/NW_315deg.jpg').unlink()
        with self.assertRaisesRegex(ValueError,'Missing actual view'):self.call()
        self.assertFalse((self.out_parent/'out').exists())

if __name__=='__main__':unittest.main()

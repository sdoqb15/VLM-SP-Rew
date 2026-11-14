"""Provenance consistency tests use synthetic JPEGs and invented provider JSON marked as fixtures."""
from pathlib import Path
import copy,json,tempfile,unittest
from PIL import Image
from annotation_cache_manifest import audit_existing_cache,DIRECTIONS,sha

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.point=self.root/'pos_0.000_0.000';self.point.mkdir()
        self.goal=self.root/'ENGINEERING_GOAL.jpg';Image.new('RGB',(4,4),(255,128,0)).save(self.goal)
        self.views=[]
        for k,d in enumerate(DIRECTIONS):
            p=self.point/(d+'_'+str(k*45)+'deg.jpg')
            Image.new('RGB',(8,8),(k*29,k*13,k*7)).save(p)
            self.views.append(sha(p.read_bytes()))
        objects='Objects: HousePlant\nRoom: Kitchen\nTarget_visible: yes'
        navigability='\n'.join('Navigable_'+d+': yes'for d in DIRECTIONS)
        (self.point/'gpt4o_analysis.txt').write_text(objects+'\n'+navigability+'\n')
        self.folders={}
        for kind,text,images in [('objects_room_visibility',objects,self.views+[sha(self.goal.read_bytes())]),('navigability',navigability,self.views)]:
            folder=self.point/'api_audit'/(kind+'_ENGINEERING_FIXTURE');folder.mkdir(parents=True)
            self.folders[kind]=folder
            request=dict(UTC='2026-11-14T00:00:00Z',kind=kind,requested_model_snapshot='ENGINEERING_ONLY_2026-11-14',image_payload_hashes=[dict(sha256=x)for x in images])
            provider=dict(id='ENGINEERING_NOT_REAL_CALL',model='ENGINEERING_ONLY_2026-11-14',usage=dict(total_tokens=0),choices=[dict(message=dict(content=text))])
            complete=dict(requested_model=request['requested_model_snapshot'],returned_model=provider['model'],text_sha256=sha(text.encode()),provider_response_id=provider['id'],usage=provider['usage'])
            for n,data in [('REQUEST.json',request),('PROVIDER_RESPONSE.json',provider),('COMPLETE.json',complete)]:
                (folder/n).write_text(json.dumps(data))
            (folder/'RESPONSE_TEXT.txt').write_text(text)
    def tearDown(self):self.tmp.cleanup()
    def audit(self,**kwargs):return audit_existing_cache(self.root,self.goal,**kwargs)
    def mutate(self,kind,name,fn):
        p=self.folders[kind]/name;data=json.loads(p.read_text());fn(data);p.write_text(json.dumps(data))
    def test_internal_consistency_only_never_proves_actual_provider_call(self):
        result=self.audit()
        self.assertTrue(result['source_integrity_qualified'])
        self.assertTrue(result['not_annotation_accuracy_or_navigation_efficacy'])
        self.assertEqual(len(result['provider_records']),2)
        self.assertEqual(len(result['points'][0]['views']),8)
    def test_missing_eighth_view_fails(self):
        (self.point/'NW_315deg.jpg').unlink()
        with self.assertRaisesRegex(ValueError,'Missing actual view'):self.audit()
    def test_swapped_image_order_is_not_equivalent(self):
        def swap(data):data['image_payload_hashes'][0],data['image_payload_hashes'][1]=data['image_payload_hashes'][1],data['image_payload_hashes'][0]
        self.mutate('navigability','REQUEST.json',swap)
        with self.assertRaisesRegex(ValueError,'image/reference hashes mismatch'):self.audit()
    def test_provider_text_tampering_fails(self):
        (self.folders['navigability']/'RESPONSE_TEXT.txt').write_text('different')
        with self.assertRaisesRegex(ValueError,'text was modified'):self.audit()
    def test_no_provider_fields_cannot_qualify(self):
        folder=self.folders['navigability'];text='No usable annotation'
        (folder/'RESPONSE_TEXT.txt').write_text(text)
        self.mutate('navigability','PROVIDER_RESPONSE.json',lambda data:data['choices'][0]['message'].update(content=text))
        self.mutate('navigability','COMPLETE.json',lambda data:data.update(text_sha256=sha(text.encode())))
        with self.assertRaisesRegex(ValueError,'lacks required fields'):self.audit()
    def test_family_alias_cannot_be_pinned_release(self):
        self.mutate('navigability','REQUEST.json',lambda data:data.update(requested_model_snapshot='gpt-4o'))
        with self.assertRaisesRegex(ValueError,'Dated annotation model snapshot'):self.audit()
    def test_provider_usage_mismatch_is_rejected(self):
        self.mutate('navigability','COMPLETE.json',lambda data:data.update(usage={'total_tokens':999}))
        with self.assertRaisesRegex(ValueError,'ID/usage'):self.audit()
    def test_complete_model_output_must_match_derived_annotation(self):
        p=self.point/'gpt4o_analysis.txt';p.write_text(p.read_text().replace('Room: Kitchen','Room: Bathroom'))
        with self.assertRaisesRegex(ValueError,'differs from raw provider response'):self.audit()
    def test_parsed_only_is_explicitly_unqualified(self):
        for folder in self.folders.values():
            (folder/'COMPLETE.json').rename(folder/'ENGINEERING_REMOVED_COMPLETE.json')
        with self.assertRaisesRegex(ValueError,'No immutable provider provenance'):self.audit()
        result=self.audit(allow_parsed_only=True)
        self.assertFalse(result['source_integrity_qualified'])
        self.assertTrue(result['parsed_only_legacy'])
if __name__=='__main__':unittest.main()

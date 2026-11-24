"""Navigation references must satisfy actual visibility and obstacle-aware paths."""
import copy,unittest
from types import SimpleNamespace
from vlm_sp_rew.qualify_paths import native_reference,reference_controller_spec

class NativeControllerFixture:
    def __init__(self):self.position=dict(x=0.,y=.9,z=0.);self.yaw=0.;self.calls=[];self.bad_endpoint=False
    def event(self,ok=True,result=None):
        return SimpleNamespace(metadata=dict(lastActionSuccess=ok,agent=dict(position=dict(self.position),rotation=dict(y=self.yaw)),
            actionReturn=result,objects=[dict(objectId='plant-fixed',objectType='HousePlant',position=dict(x=2.,y=.9,z=0.),visible=self.position['x']==2.)]))
    def step(self,**kw):
        self.calls.append(kw);a=kw['action']
        if a=='Teleport':self.position=dict(kw['position']);self.yaw=kw['rotation']['y'];return self.event()
        if a in ('LookUp','LookDown'):return self.event(False)
        if a=='GetReachablePositions':return self.event(result=[dict(x=i,y=.9,z=0.) for i in (0.,1.,2.)])
        if a=='GetShortestPathToPoint':
            corners=[dict(x=0.,y=.9,z=0.),dict(x=0.,y=.9,z=2.),dict(x=2.,y=.9,z=2.),dict(x=2.,y=.9,z=0.)]
            if self.bad_endpoint:corners[-1]['x']=1.9
            return self.event(result=dict(corners=corners))
        raise AssertionError(a)

class NativePathTests(unittest.TestCase):
    def test_continuous_reference_oracle_does_not_change_the_policy_controller(self):
        policy=dict(controller_kwargs=dict(snapToGrid=True,gridSize=.25,rotateStepDegrees=90))
        oracle=reference_controller_spec(policy)
        self.assertTrue(policy['controller_kwargs']['snapToGrid'])
        self.assertFalse(oracle['controller_kwargs']['snapToGrid'])
        self.assertEqual(oracle['controller_kwargs']['gridSize'],.25)
        habitat=dict(backend='habitat',gpu_device_id=0)
        self.assertEqual(reference_controller_spec(habitat),habitat)
    def episode(self):return dict(goal_instance_ids=['plant-fixed'],goal_category='HousePlant',start_position=dict(x=0.,y=.9,z=0.),start_yaw_degrees=0.,start_horizon_degrees=0.)
    def test_occluded_near_pose_is_not_a_success_region_and_path_is_not_euclidean(self):
        c=NativeControllerFixture();r=native_reference(c,self.episode(),'ProcTHOR',c.step)
        self.assertEqual(r['shortest_path_m'],6.)
        self.assertEqual(r['eligible_positions'],1);self.assertEqual(r['endpoint']['x'],2.)
        self.assertTrue(any(x['action']=='LookUp'for x in c.calls))
    def test_four_action_r2r_uses_native_viewpoints_without_look_queries(self):
        c=NativeControllerFixture();r=native_reference(c,self.episode(),'R2R-adapted',c.step,[dict(x=1.,y=.9,z=0.),dict(x=2.,y=.9,z=0.)])
        self.assertEqual(r['shortest_path_m'],6.)
        self.assertFalse(any(x['action'] in ('LookUp','LookDown','GetReachablePositions') for x in c.calls))
    def test_r2r_never_uses_random_navigable_samples_as_complete_goal_region(self):
        c=NativeControllerFixture()
        with self.assertRaisesRegex(ValueError,'native ObjectNav'):native_reference(c,self.episode(),'R2R-adapted',c.step)
    def test_initial_success_has_zero_length_reference(self):
        c=NativeControllerFixture();e=self.episode();e['start_position']['x']=2.
        r=native_reference(c,e,'ProcTHOR',c.step);self.assertEqual(r['shortest_path_m'],0.)
    def test_visible_after_rotation_at_start_still_has_zero_translation_reference(self):
        class TurnOnly(NativeControllerFixture):
            def event(self,*a,**kw):
                result=super().event(*a,**kw)
                result.metadata['objects'][0]['visible']=self.position['x']==2. and self.yaw==90.
                return result
        c=TurnOnly();e=self.episode();e['start_position']['x']=2.
        result=native_reference(c,e,'ProcTHOR',c.step)
        self.assertEqual(result['shortest_path_m'],0.)
        self.assertFalse(any(x['action']=='GetShortestPathToPoint'for x in c.calls))
    def test_shortest_path_endpoint_mismatch_is_rejected(self):
        c=NativeControllerFixture();c.bad_endpoint=True
        with self.assertRaisesRegex(ValueError,'endpoints'):native_reference(c,self.episode(),'ProcTHOR',c.step)
    def test_native_projected_endpoint_must_be_observed_in_the_same_success_region(self):
        class Projected(NativeControllerFixture):
            def event(self,*args,**kwargs):
                result=super().event(*args,**kwargs)
                result.metadata['objects'][0]['visible']=self.position['x']>=1.9
                return result
        c=Projected();c.bad_endpoint=True
        result=native_reference(c,self.episode(),'ProcTHOR',c.step)
        self.assertEqual(result['endpoint']['x'],1.9)
        self.assertTrue(result['native_projection_verified'])
        self.assertEqual(result['visible_pose']['position']['x'],1.9)
        self.assertGreater(result['shortest_path_m'],6.)
    def test_partial_native_path_cannot_supply_a_reference(self):
        class Partial(NativeControllerFixture):
            def step(self,**kwargs):
                result=super().step(**kwargs)
                if kwargs['action']=='GetShortestPathToPoint':result.metadata['actionReturn']['status']='PathPartial'
                return result
        c=Partial()
        with self.assertRaisesRegex(ValueError,'No complete native path'):native_reference(c,self.episode(),'ProcTHOR',c.step)
    def test_fixed_target_is_never_replaced_by_another_instance(self):
        c=NativeControllerFixture();e=self.episode();e['goal_instance_ids']=['not-the-actual-object']
        with self.assertRaisesRegex(ValueError,'requested goal absent'):native_reference(c,e,'ProcTHOR',c.step)

if __name__=='__main__':unittest.main()

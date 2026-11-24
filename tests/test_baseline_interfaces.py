"""Cross-framework observation/task contracts must not add privileged policy inputs."""
from pathlib import Path
import json,os,subprocess,sys,unittest
from types import SimpleNamespace
import numpy as np
from vlm_sp_rew.environment import spaces
from vlm_sp_rew.baselines.common import ArraySpec,task_contract,check_task_contract
from vlm_sp_rew.baselines.dynalang import decoder_300

class BaselineInterfaces(unittest.TestCase):
    def test_native_array_spec_has_the_same_task_shapes_without_importing_a_learner(self):
        for benchmark,n in [('RoboTHOR',6),('ProcTHOR',6),('R2R-adapted',4)]:
            obs,act=spaces(['HousePlant'],benchmark,ArraySpec)
            self.assertEqual(obs['image'].shape,(300,300,3));self.assertEqual(obs['image'].dtype,np.dtype(np.uint8))
            self.assertEqual(int(act['action'].high),n);self.assertNotIn('goal',obs)
    def test_bare_native_environment_import_does_not_initialize_jax_or_elements(self):
        env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).parents[1]/'src')
        result=subprocess.run([sys.executable,'-c',"import vlm_sp_rew.environment,sys;assert 'jax'not in sys.modules;assert 'elements'not in sys.modules"],env=env,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_checkpoint_rejects_other_goal_or_action_order(self):
        env=SimpleNamespace(benchmark='ProcTHOR',actions=('MoveAhead','RotateLeft','RotateRight','LookUp','LookDown','Stop'),vocabulary=['HousePlant'])
        contract=task_contract(env);check_task_contract(contract,env)
        wrong=json.loads(json.dumps(contract));wrong['goal_vocabulary']=['Mug']
        with self.assertRaisesRegex(ValueError,'contract mismatch'):check_task_contract(wrong,env)
        wrong=json.loads(json.dumps(contract));wrong['actions'][3],wrong['actions'][5]=wrong['actions'][5],wrong['actions'][3]
        with self.assertRaisesRegex(ValueError,'contract mismatch'):check_task_contract(wrong,env)
    def test_r2d2_native_convolution_receives_normalized_float_pixels(self):
        from vlm_sp_rew.baselines.r2d2 import normalized_rgb
        image=np.full((300,300,3),255,np.uint8);image[0,0]=0
        result=normalized_rgb(image)
        self.assertEqual(result.dtype,np.dtype(np.float32))
        self.assertEqual(result.shape,image.shape)
        self.assertEqual(float(result[0,0,0]),0.)
        self.assertEqual(float(result[-1,-1,0]),1.)
        self.assertEqual(int(image[-1,-1,0]),255)
        with self.assertRaises(ValueError):normalized_rgb(image.astype(np.float32))
    def test_benchmark_labels_cannot_silently_select_another_simulator(self):
        from vlm_sp_rew.reward import validate_scene_backend
        house=dict(rooms=[dict(floorPolygon=[])])
        validate_scene_backend(house,'ProcTHOR')
        validate_scene_backend(dict(backend='robothor'),'RoboTHOR')
        validate_scene_backend(dict(backend='habitat'),'R2R-adapted')
        for benchmark,scene in [('RoboTHOR',house),('R2R-adapted',house),('ProcTHOR',dict(backend='habitat',rooms=[{}]))]:
            with self.assertRaises(ValueError):validate_scene_backend(scene,benchmark)
    def test_dynalang_stride_adapter_preserves300_reconstruction_and_encoder_interface(self):
        class ShapeDecoder:
            def __init__(self,shape,*args,**kw):self.shape=shape;self.minres=kw['minres']
            def __call__(self,x):return np.ones((x.shape[0],*self.shape),np.float32)
        nets=SimpleNamespace(ImageDecoderResnet=ShapeDecoder)
        original=decoder_300(nets)
        decoder=nets.ImageDecoderResnet((300,300,3),depth=8,blocks=0,resize='stride',minres=4,sigmoid=False)
        self.assertEqual(decoder.shape,(320,320,3));self.assertEqual(decoder.minres,5)
        self.assertEqual(decoder(np.zeros((1,4))).shape,(1,300,300,3));self.assertIs(original,ShapeDecoder)

if __name__=='__main__':unittest.main()

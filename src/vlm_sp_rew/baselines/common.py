"""Array-space metadata for adapters running in third-party learner environments."""
import numpy as np

class ArraySpec:
    def __init__(self,dtype,shape=(),low=None,high=None):
        self.dtype=np.dtype(dtype);self.shape=tuple(shape)
        if self.dtype==np.dtype(bool):lo,hi=False,True
        elif np.issubdtype(self.dtype,np.integer):info=np.iinfo(self.dtype);lo,hi=info.min,info.max
        else:lo,hi=-np.inf,np.inf
        self.low=np.asarray(lo if low is None else low,dtype=self.dtype)
        self.high=np.asarray(hi if high is None else high,dtype=self.dtype)
    @property
    def discrete(self):return np.issubdtype(self.dtype,np.integer) or self.dtype==np.dtype(bool)


def task_contract(env):
    return dict(benchmark=env.benchmark,actions=list(env.actions),goal_vocabulary=list(env.vocabulary),
                RGB=[300,300,3],goal_channel=False,max_episode_actions=200)


def check_task_contract(saved,env):
    if saved!=task_contract(env):raise ValueError('Checkpoint benchmark, fixed goal or deployed observation/action contract mismatch')


def configure_local_ipc(root=None):
    """Private short Unix-socket directory, independent of long dataset/TMP paths."""
    import multiprocessing as mp
    import multiprocessing.util
    from pathlib import Path
    import os,tempfile,shutil
    base=Path(root or '/tmp');base.mkdir(parents=True,exist_ok=True)
    if len(os.fsencode(str(base.resolve())))>65:raise ValueError('Use a short local --ipc-root for multiprocessing Unix sockets')
    directory=tempfile.mkdtemp(prefix='vs-',dir=base)
    os.chmod(directory,0o700)
    mp.current_process()._config['tempdir']=directory
    multiprocessing.util.Finalize(None,shutil.rmtree,args=(directory,),exitpriority=-100)
    return directory

"""Prepare a single-process Acme source copy with native learner code intact."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile


SIGNALS='''import contextlib
import signal
import threading


@contextlib.contextmanager
def runtime_terminator(callback=None):
    """Use operating-system termination signals for the local runner."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('The single-process Acme loop runs on the main thread')
    def stop(signum, frame):
        if callback is not None:
            callback()
        raise SystemExit(128 + signum)
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        for number in previous:
            signal.signal(number, stop)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
'''


def prepare_local_source(source,destination,revision):
    """Keep the original checkout; isolate local-runtime compatibility edits."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:raise ValueError('Acme revision mismatch')
    subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD','--','acme'],check=True,capture_output=True)
    files={p.relative_to(source).as_posix():p.read_bytes()for p in (source/'acme').rglob('*')
        if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'}
    for name in ('LICENSE','LICENSE.txt','NOTICE'):
        if (source/name).is_file():files[name]=(source/name).read_bytes()
    signals='acme/utils/signals.py';original=files[signals].decode()
    if original.count('import launchpad')!=1:raise ValueError('Unexpected native signal module')
    # Retain the author's copyright/license header in the derived module.
    files[signals]=(original[:original.index('"""Helper methods')]+SIGNALS).encode()
    init='acme/jax/experiments/__init__.py';text=files[init].decode()
    for name in ('make_distributed_experiment','make_distributed_offline_experiment'):
        line='from acme.jax.experiments.%s import %s\n'%(name,name)
        if text.count(line)!=1:raise ValueError('Unexpected native experiment exports')
        text=text.replace(line,'')
    files[init]=text.encode()
    utils='acme/jax/utils.py';text=files[utils].decode()
    if text.count('jax.P(')!=3:raise ValueError('Unexpected native device-sharding helpers')
    # The aliases refer to the same native sharding classes on JAX0.4.33.
    text=text.replace('jax.NamedSharding(', 'jax.sharding.NamedSharding(')
    files[utils]=text.replace('jax.P(', 'jax.sharding.PartitionSpec(').encode()
    hashes={name:hashlib.sha256(raw).hexdigest()for name,raw in sorted(files.items())}
    receipt=dict(source_revision=revision,compatibility='single-process signal handlers, local experiment exports and fully qualified native JAX sharding names',
        changed_files=[signals,init,utils],files_sha256=hashes)
    root=destination/('acme-local-'+revision)
    if root.exists():
        if json.loads((root/'SOURCE_RECEIPT.json').read_text())!=receipt:raise ValueError('Existing local Acme source has a different receipt')
        for name,digest in hashes.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Local Acme source changed: '+name)
        return root
    destination.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination).free-sum(map(len,files.values()))<110*1024**3:
        raise RuntimeError('Local author-source copy would exhaust the operational storage margin')
    # The generated copy contains author code and its license, never fake modules.
    temporary=Path(tempfile.mkdtemp(prefix='acme-local-',dir=destination))
    for name,raw in files.items():
        path=temporary/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (temporary/'SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    temporary.rename(root)
    return root

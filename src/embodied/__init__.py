__version__ = '2.0.0'

try:
  import colored_traceback
  colored_traceback.add_hook(colors='terminal')
except ImportError:
  pass

from .core import *

from . import envs
from . import run


def __getattr__(name):
  # Native simulator and external-baseline interfaces do not initialize JAX.
  if name == 'jax':
    import importlib
    value = importlib.import_module('.jax', __name__)
    globals()['jax'] = value
    return value
  raise AttributeError(name)

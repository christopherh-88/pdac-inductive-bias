import importlib.util
import multiprocessing as mp
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'download' / 'kaggle_stream_cohort.py'


sys.modules.setdefault('SimpleITK', types.ModuleType('SimpleITK'))
sys.modules.setdefault('stream_unzip', types.ModuleType('stream_unzip'))

spec = importlib.util.spec_from_file_location('kaggle_stream_cohort', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_make_and_reset_worker_pool():
    ctx = mp.get_context('spawn')
    pool = module.make_worker_pool(ctx)
    assert pool is not None

    pool = module.reset_worker_pool(pool, ctx)
    assert pool is not None
    pool.shutdown(wait=True, cancel_futures=True)

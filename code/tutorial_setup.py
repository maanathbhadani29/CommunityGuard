"""Visible, ordinary setup helpers for the Python 3.13 teaching project.

No archive decoder, custom cell magic, secondary Python, or science worker.
All lessons execute in the user's current kernel. pip installs are the only
subprocesses used by the default tutorial.
"""
from pathlib import Path
import contextlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time


def installed_version(name):
    try:
        return importlib.metadata.version(name).split('+')[0]
    except importlib.metadata.PackageNotFoundError:
        return None


def install_command(command, folder, label, timeout=900):
    log_path = Path(folder)/'setup_install.log'
    print(label, flush=True)
    print('Command:', subprocess.list2cmdline(command), flush=True)
    print('Full pip output:', log_path, flush=True)
    started = time.monotonic()
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed > timeout:
                    raise TimeoutError(f'{label} exceeded {timeout//60} minutes. See {log_path}; check the connection and rerun setup.')
                lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
                latest = next((line for line in reversed(lines) if line.strip()), 'Waiting for pip...')
                print(f'  {elapsed:.0f}s | {latest[-180:]}', flush=True)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                tail = log_path.read_text(encoding='utf-8', errors='replace')[-5000:]
                raise RuntimeError(f'{label} failed; no experiment has started.\n{tail}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    print(f'{label} finished in {time.monotonic()-started:.1f}s.', flush=True)


def ensure_packages(root):
    root = Path(root)
    if not (3,10) <= sys.version_info[:2] <= (3,13):
        raise RuntimeError('Use a Python 3.13 kernel for this tested tutorial. The pinned wheels cover Python 3.10–3.13.')
    required_file = root/'code/requirements-python313.txt'
    required = dict(line.strip().split('==',1) for line in required_file.read_text().splitlines()
                    if '==' in line and not line.lstrip().startswith('#'))
    required['torch'] = '2.6.0'
    print('Python:', sys.version.split()[0], '| executable:', sys.executable, flush=True)
    print('Requirements file:', required_file, flush=True)
    print(json.dumps(required, indent=2), flush=True)
    for module, distribution in {'numpy':'numpy','pandas':'pandas','scipy':'scipy','sklearn':'scikit-learn','torch':'torch','matplotlib':'matplotlib'}.items():
        if module in sys.modules and installed_version(distribution) != required[distribution]:
            raise RuntimeError(f'{module} is already loaded with a different version. Restart the kernel and run setup before importing numerical packages.')
    pip = [sys.executable,'-u','-m','pip','install','--disable-pip-version-check',
           '--no-input','--only-binary=:all:','--progress-bar','off','--timeout','25','--retries','1']
    if installed_version('torch') != required['torch']:
        install_command(pip+['--index-url','https://download.pytorch.org/whl/cpu','torch==2.6.0'],root,'Install CPU PyTorch')
    if any(installed_version(k) != v for k,v in required.items() if k != 'torch'):
        install_command(pip+['-r',str(required_file)],root,'Install scientific packages')
    importlib.invalidate_caches()
    mismatches = {k:installed_version(k) for k,v in required.items() if installed_version(k) != v}
    if mismatches:
        raise RuntimeError(f'Package versions do not match: {mismatches}. Inspect setup_install.log.')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy, pandas, sklearn, torch, matplotlib
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    limits = threadpool_limits(limits=1)
    print('SETUP COMPLETE. Continue with the ordinary Python lesson cells.', flush=True)
    return limits


@contextlib.contextmanager
def progress(label):
    """Print elapsed time; calculations stay in the notebook's main thread."""
    started = time.monotonic()
    done = threading.Event()
    print(label + ' — started', flush=True)
    def report():
        while not done.wait(10):
            print(f'  {label}: {time.monotonic()-started:.0f}s elapsed', flush=True)
    thread = threading.Thread(target=report,daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        print(label + ' — interrupted or failed; see the error below',flush=True)
        raise
    else:
        print(f'{label} — finished in {time.monotonic()-started:.1f}s',flush=True)
    finally:
        done.set()
        thread.join(timeout=1)

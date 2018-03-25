# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <name> <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --download               Download data.
    --source-only            Process source only.
    --target-only            Process target only.
    -h, --help               Show help message.
"""

from docopt import docopt
from multiprocessing import cpu_count
import importlib
from data import SOURCE_ONLY, TARGET_ONLY, SOURCE_AND_TARGET

if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() if num_workers is None else int(num_workers)
    download = args["--download"]
    source_only = args["--source-only"]
    target_only = args["--target-only"]
    mode = SOURCE_AND_TARGET
    if source_only:
        mode = SOURCE_ONLY
    if target_only:
        mode = TARGET_ONLY

    assert name in ["jsut"]
    mod = importlib.import_module("data." + name)
    instance = mod.instantiate(in_dir, out_dir)
    if download:
        instance.download()
    instance.preprocess(num_workers, mode=mode)
# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <name> <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --download=true          Download data.
    -h, --help               Show help message.
"""

from docopt import docopt
from multiprocessing import cpu_count
import importlib

if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() if num_workers is None else int(num_workers)
    download = args["--download"]
    download = download == "1" or download == "true"

    assert name in ["jsut"]
    mod = importlib.import_module("data." + name)
    instance = mod.instantiate(in_dir, out_dir)
    if download:
        instance.download()
    instance.preprocess(num_workers)
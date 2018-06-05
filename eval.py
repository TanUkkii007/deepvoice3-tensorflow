"""Trainining script for seq2seq text-to-speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --dataset=<name>             Dataset name.
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --checkpoint-seq2seq=<path>  Restore seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>  Restore postnet model from checkpoint path.
    -h, --help                   Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import importlib
from deepvoice3.frontend import Frontend
from deepvoice3.models import SingleSpeakerTTSModel
from hparams import hparams, hparams_debug_string


def eval(hparams, model_dir, source_files, target_files, checkpoint_path=None):
    def eval_input_fn():
        source = tf.data.TFRecordDataset(list(source_files)[:16])
        target = tf.data.TFRecordDataset(list(target_files)[:16])

        frontend = Frontend(source, target, hparams)
        batched = frontend.prepare(

        ).zip_source_and_target(

        )
        batched = batched.swap_source() if hparams.swap_source else batched
        batched = batched.group_by_batch(

        ).add_memory_mask(

        ).add_frame_positions(

        ).add_target_mask(

        ).downsample_mel(

        ).dataset
        return batched

    estimator = SingleSpeakerTTSModel(hparams, model_dir)

    eval_metrics = estimator.evaluate(lambda: eval_input_fn(), checkpoint_path=checkpoint_path)


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint-seq2seq"]
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["jsut"]
    dataset = importlib.import_module("data." + dataset_name)
    dataset_instance = dataset.instantiate(in_dir="", out_dir=data_root)

    hparams.parse(args["--hparams"])
    hparams.batch_size = 1
    print(hparams_debug_string())

    tf.logging.set_verbosity(tf.logging.INFO)
    eval(hparams, checkpoint_dir, dataset_instance.source_files, dataset_instance.target_files, checkpoint_path)


if __name__ == '__main__':
    main()
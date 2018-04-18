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
    --train-seq2seq-only         Train only seq2seq model.
    --train-postnet-only         Train only postnet model.
    -h, --help                   Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import importlib
from deepvoice3_tensorflow.frontend import Frontend
from deepvoice3_tensorflow.models import SingleSpeakerTTSModel
from hparams import hparams, hparams_debug_string

def train(hparams, model_dir, source_files, target_files):
    def train_input_fn():
        source = tf.data.TFRecordDataset(list(source_files))
        target = tf.data.TFRecordDataset(list(target_files))

        frontend = Frontend(source, target, hparams)
        batched = frontend.prepare(

        ).zip_source_and_target(

        ).repeat(

        ).shuffle(
            buffer_size=hparams.batch_size*10
        ).group_by_batch(

        ).swap_source_random(
            swap_probability=hparams.replace_pronunciation_prob
        ).add_memory_mask(

        ).add_frame_positions(

        ).add_target_mask(

        ).downsample_mel(

        ).dataset
        return batched

    run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps, log_step_count_steps=hparams.log_step_count_steps)
    estimator = SingleSpeakerTTSModel(hparams, model_dir, config=run_config)

    estimator.train(lambda: train_input_fn())



def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["jsut"]
    dataset = importlib.import_module("data." + dataset_name)
    dataset_instance = dataset.instantiate(in_dir="", out_dir=data_root)

    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    tf.logging.set_verbosity(tf.logging.INFO)
    train(hparams, checkpoint_dir, dataset_instance.source_files, dataset_instance.target_files)


if __name__ == '__main__':
    main()
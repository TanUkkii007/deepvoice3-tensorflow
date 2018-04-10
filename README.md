# deepvoice3-tensorflow

A tensorflow based implementation of [DeepVoice3](https://arxiv.org/abs/1710.07654).

This project is ported from [the pytorch based DeepVoice3 implementation](https://github.com/r9y9/deepvoice3_pytorch) created by @r9r9.


## Status

This project is currently work in progress.

My goal is to build a Japanese end-to-end TTS model by using DeepVoice3.
If you are interested in multi-speaker implementation with various dataset support and pre-trained models, please refer to the original implementation: https://github.com/r9y9/deepvoice3_pytorch.

Current limitations of this project are following:

- Only [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) dataset is supported
- No multi-speaker implementation
- Evaluation and inference scripts are not ready
- Mel-to-linear spectrogram converter is not implemented
- Training and hyper parameter tuning is ongoing

I still have not obtained distinct and monotonic alignments.
I will report once I am able to get a good result.


## Requirements

- python >= 3.6
- tensorflow >= 1.7


## Installation

```
pip install -e ".[train]"
pip install -e ".[test]"
pip install -e ".[jp]"
```


## Preprocessing

The following command preprocesses text and audio data. The name argument must be jsut since JSUT is the only dataset that is supported now.

```
preprocess.py [options] <name> <in-dir> <out-dir>
```

You can specify `--download` option to download the dataset.

```
preprocess.py --download jsut <path-to-download-dataset> <path-to-output-preprocessed-data>
```

## Training

```
python train.py --checkpoint-dir=<path-to-checkpoint-dir> --data-root=<path-to-preprocessed-data> --dataset=jsut
```

## Visualizing alignments

At training time, a TFRecord file that contains alignment information is generated per certain timesteps.
You can visualize alignments between source and target by specifying the TFRecord file.

```
python visualize_alignment.py <tfrecord-file-name>
```

## Running tests

```
bazel test tests:all
```


## Acknowledgements

Since I am new to this area, I am learning an end-to-end TTS model with implementations that are publicly available.
The following implementations are very helpful and I appreciate the authors of these implementations.
I hope my work is helpful for someone who is interested in an end-to-end TTS model as well.

- https://github.com/r9y9/deepvoice3_pytorch
- https://github.com/keithito/tacotron
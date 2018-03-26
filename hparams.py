import tensorflow as tf


hparams = tf.contrib.training.HParams(
    name="deepvoice3",
    frontend="en",

    # Audio
    num_mels=80,
    fmin=125,
    fmax=7600,
    fft_size=1024,
    hop_size=256,
    sample_rate=22050,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    rescaling=False,
    rescaling_max=0.999,
    allow_clipping_in_normalization=False,

    # Training
    batch_size=16,
    approx_min_target_length = 200,
    batch_bucket_width = 20,
    batch_num_buckets = 20,
    )


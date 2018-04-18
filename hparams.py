import tensorflow as tf


hparams = tf.contrib.training.HParams(
    name="deepvoice3",

    # Text
    replace_pronunciation_prob=0.5,

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

    # Model:
    downsample_step=4,
    outputs_per_step=3,
    embedding_weight_std=0.1,
    padding_idx=0,
    # Maximum number of input text length
    # try setting larger value if you want to give very long text input
    max_positions=512,
    n_vocab=0xffff, # jsut
    dropout=1 - 0.95,
    kernel_size=3,
    text_embed_dim=128,
    text_embedding_weight_std=0.1,
    encoder_channels=256,
    decoder_channels=256,
    query_position_rate=1.0,
    max_decoder_steps=200,
    min_decoder_steps=10,
    # can be computed by `compute_timestamp_ratio.py`.
    key_position_rate=1.03, # for jsut
    use_memory_mask=True,
    trainable_positional_encodings=False,
    freeze_embedding=False,

    # Training
    batch_size=16,
    approx_min_target_length=200,
    batch_bucket_width=40,
    batch_num_buckets=50,
    initial_learning_rate=1e-4,  # 0.0001,
    adam_beta1=0.5,
    adam_beta2=0.9,
    adam_eps=1e-6,
    save_summary_steps=50,
    log_step_count_steps=1,
    alignment_save_steps=100,
    )


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
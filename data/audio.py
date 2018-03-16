import librosa
from hparams import hparams
import lws
import numpy as np


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]

def preemphasis(x):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x, hparams.preemphasis)

def spectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def melspectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    if not hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    return _normalize(S)


def _lws_processor():
    return lws.lws(hparams.fft_size, hparams.hop_size, mode="speech")


def _build_mel_basis():
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.fft_size, fmin=hparams.fmin, fmax=hparams.fmax, n_mels=hparams.num_mels)

_mel_basis = _build_mel_basis()

def _linear_to_mel(spectrogram):
    return np.dot(_mel_basis, spectrogram)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x + 0.01))

def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

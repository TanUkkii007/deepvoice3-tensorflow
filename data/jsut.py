import os
from functools import partial
from urllib.request import urlretrieve
from tqdm import tqdm
import zipfile
import data.audio as audio
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from nnmnkwii.datasets import jsut
from nnmnkwii.io import hts
from hparams import hparams
import tensorflow as tf


# https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def _process_utterance(out_dir, index, wav_path, text):
    sr = hparams.sample_rate
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    lab_path = wav_path.replace("wav/", "lab/").replace(".wav", ".lab")

    # Trim silence from hts labels if available
    if os.path.exists(lab_path):
        labels = hts.load(lab_path)
        assert labels[0][-1] == "silB"
        assert labels[-1][-1] == "silE"
        begin = int(labels[0][1] * 1e-7 * sr)
        end = int(labels[-1][0] * 1e-7 * sr)
        wav = wav[begin:end]
    else:
        wav, _ = librosa.effects.trim(wav, top_db=30)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'jsut-spec-%05d.npy' % index
    mel_filename = 'jsut-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text)


class JSUT():
    def __init__(self, in_dir, out_dir):
        self.dl_dir = in_dir
        self.out_dir = out_dir
        self.url = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
        self.file_name = "jsut_ver1.1.zip"
        self.file_path = os.path.join(self.dl_dir, self.file_name)

    def download(self):
        if not os.path.exists(self.dl_dir):
            os.makedirs(self.dl_dir)
        if not os.path.exists(self.file_path):
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                          desc="downloading " + self.file_name) as t:  # all optional kwargs
                urlretrieve(self.url, filename=self.file_path, reporthook=t.update_to,
                            data=None)
        if not os.path.exists(self.in_dir) and os.path.exists(self.file_path):
            with zipfile.ZipFile(self.file_path, "r") as zip_ref:
                members = tqdm(zip_ref.namelist(), desc="unzipping " + self.file_name)
                for zipinfo in members:
                    zip_ref.extract(zipinfo, self.dl_dir)

    def preprocess(self, num_workers=4):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = []
        transcriptions = jsut.TranscriptionDataSource(
            self.in_dir, subsets=jsut.available_subsets).collect_files()
        wav_paths = jsut.WavFileDataSource(
            self.in_dir, subsets=jsut.available_subsets).collect_files()

        for index, (text, wav_path) in enumerate(zip(transcriptions, wav_paths)):
            futures.append(executor.submit(partial(_process_utterance, self.out_dir, index + 1, wav_path, text)))
        result = [future.result() for future in tqdm(futures)]
        executor.shutdown()
        self._write_metadata(result)

    def _write_metadata(self, metadata):
        with open(os.path.join(self.out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            for m in metadata:
                f.write('|'.join([str(x) for x in m]) + '\n')
        frames = sum([m[2] for m in metadata])
        frame_shift_ms = hparams.hop_size / hparams.sample_rate * 1000
        hours = frames * frame_shift_ms / (3600 * 1000)
        print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
        print('Max input length:  %d' % max(len(m[3]) for m in metadata))
        print('Max output length: %d' % max(m[2] for m in metadata))


    @property
    def in_dir(self):
        return self.file_path.strip(".zip")
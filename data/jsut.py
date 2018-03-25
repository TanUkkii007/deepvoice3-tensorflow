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
from data.tfrecord_utils import write_preprocessed_target_data, write_preprocessed_source_data2
from data import TqdmUpTo, TargetMetaData, SourceMetaData, SOURCE_AND_TARGET, SOURCE_ONLY, TARGET_ONLY
from janome.tokenizer import Tokenizer
import jaconv
from typing import List

n_vocab = 0xffff
_eos = 1
_tokenizer = Tokenizer()


def _yomi(result):
    tokens = []
    yomis = []
    for token in result[:-1]:
        tokens.append(token.surface)
        yomi = None if token.phonetic == "*" else token.phonetic
        yomis.append(yomi)
    return tokens, yomis


def _mix_pronunciation(tokens, yomis):
    return "".join(
        yomis[idx] if yomis[idx] is not None else tokens[idx]
        for idx in range(len(tokens))
    )


def mix_pronunciation(text):
    tokens, yomis = _yomi(_tokenizer.tokenize(text))
    return _mix_pronunciation(tokens, yomis)


def add_punctuation(text):
    last = text[-1]
    if last not in [".", ",", "、", "。", "！", "？", "!", "?"]:
        text = text + "。"
    return text


def normalize_delimiter(text):
    text = text.replace(",", "、")
    text = text.replace(".", "。")
    text = text.replace("，", "、")
    text = text.replace("．", "。")
    return text


def text_to_sequence(text, mix=False):
    for c in [" ", "　", "「", "」", "『", "』", "・", "【", "】",
              "（", "）", "(", ")"]:
        text = text.replace(c, "")
    text = text.replace("!", "！")
    text = text.replace("?", "？")

    text = normalize_delimiter(text)
    text = jaconv.normalize(text)
    if mix:
        text = mix_pronunciation(text)
    text = jaconv.hira2kata(text)
    text = add_punctuation(text)

    return [ord(c) for c in text] + [_eos], text


def sequence_to_text(seq):
    return "".join(chr(n) for n in seq)


def _process_text(out_dir, index, text):
    sequence, text1 = text_to_sequence(text, mix=False)
    sequence = np.array(sequence)
    sequence_mixed, text2 = text_to_sequence(text, mix=True)
    sequence_mixed = np.array(sequence_mixed)
    filename = 'jsut-source-%05d.tfrecords' % index
    write_preprocessed_source_data2(index, text1, sequence, text2, sequence_mixed, os.path.join(out_dir, filename))
    return SourceMetaData(index, filename, text1, len(text1), len(sequence), text2, len(text2), len(sequence_mixed))


def _process_audio(out_dir, index, wav_path):
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
    filename = 'jsut-target-%05d.tfrecords' % index
    write_preprocessed_target_data(index, spectrogram.T, mel_spectrogram.T, os.path.join(out_dir, filename))

    # Return a tuple describing this training example:
    return TargetMetaData(index, filename, n_frames)


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

    def preprocess(self, num_workers=4, mode=SOURCE_AND_TARGET):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        executor = ProcessPoolExecutor(max_workers=num_workers)
        if mode in [TARGET_ONLY, SOURCE_AND_TARGET]:
            futures = []
            wav_paths = jsut.WavFileDataSource(
                self.in_dir, subsets=jsut.available_subsets).collect_files()

            for index, wav_path in enumerate(wav_paths):
                futures.append(executor.submit(partial(_process_audio, self.out_dir, index + 1, wav_path)))
            result = [future.result() for future in tqdm(futures, desc="targets")]
            self._write_target_metadata(result)
        if mode in [SOURCE_ONLY, SOURCE_AND_TARGET]:
            futures = []
            transcriptions = jsut.TranscriptionDataSource(
                self.in_dir, subsets=jsut.available_subsets).collect_files()
            for index, text in enumerate(transcriptions):
                futures.append(executor.submit(partial(_process_text, self.out_dir, index + 1, text)))
            result = [future.result() for future in tqdm(futures, desc="sources")]
            self._write_source_metadata(result)
        executor.shutdown()

    def _write_target_metadata(self, metadata: List[TargetMetaData]):
        with open(os.path.join(self.out_dir, 'train-target.txt'), 'w', encoding='utf-8') as f:
            for m in metadata:
                f.write('|'.join([str(x) for x in m]) + '\n')
        frames = sum([m.n_frames for m in metadata])
        frame_shift_ms = hparams.hop_size / hparams.sample_rate * 1000
        hours = frames * frame_shift_ms / (3600 * 1000)
        print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
        print('Max output length: %d' % max(m.n_frames for m in metadata))


    def _write_source_metadata(self, metadata: List[SourceMetaData]):
        with open(os.path.join(self.out_dir, 'train-source.txt'), 'w', encoding='utf-8') as f:
            for m in metadata:
                f.write('|'.join([str(x) for x in m]) + '\n')
        print('Max input text length:  %d' % max(m.text_length for m in metadata))
        print('Max input array length:  %d' % max(m.source_length for m in metadata))
        print('Max alternative input text length:  %d' % max(m.text2_length for m in metadata))
        print('Max alternative input array length:  %d' % max(m.source2_length for m in metadata))

    @property
    def in_dir(self):
        return self.file_path.strip(".zip")


def instantiate(in_dir, out_dir):
    return JSUT(in_dir, out_dir)

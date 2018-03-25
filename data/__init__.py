import collections
from tqdm import tqdm

SOURCE_ONLY = "SOURCE_ONLY"
TARGET_ONLY = "TARGET_ONLY"
SOURCE_AND_TARGET = "SOURCE_AND_TARGET"

class PreprocessedSourceData(collections.namedtuple("PreprocessedSourceData", ["id", "text", "source", "source_length", "text2", "source2", "source_length2"])):
    pass

class PreprocessedTargetData(collections.namedtuple("PreprocessedTargetData", ["id", "spec", "spec_width", "mel", "mel_width", "target_length"])):
    pass

class TargetMetaData(collections.namedtuple("TargetMetaData", ["id", "filename", "n_frames"])):
    pass

class SourceMetaData(collections.namedtuple("SourceMetaData", ["id", "filename", "text", "text_length", "source_length", "text2", "text2_length", "source2_length"])):
    pass

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

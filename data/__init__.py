import collections

class PreprocessedData(collections.namedtuple("PreprocessedData", ["text", "spec", "spec_width", "mel", "mel_width", "target_length"])):
    pass
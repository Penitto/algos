import numpy as np
from typing import List
from collections import defaultdict


class TF_IDF_Transformer:

    def __init__(self):
        self.idf_dict = defaultdict(int)

    def fit(self, corpus: List[str]):
        self.power = len(corpus)
        for text in corpus:
            for word in set(text.split()):
                self.idf_dict[word] += 1

    def transform_text(self, text: str) -> List[float]:
        tf = defaultdict(int)
        words = text.split()
        tf_idf = [0] * len(words)

        # Counting TF (Term Frequency)
        for word in words:
            tf[word] += 1

        for key, word in enumerate(words):
            tf_idf[key] = tf[word] / len(words) * (np.log((self.power) / (self.idf_dict[word] + 1)) + 1)

        return tf_idf


    def transform_corpus(self, corpus: List[str]) -> List[List[float]]:
        result = []
        for text in corpus:
          result.append(self.transform_text(text))
        return result

from typing import Any, List

from nltk.tokenize import word_tokenize  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
from nltk.corpus import stopwords  # type: ignore

import math


class TextProcessing:
    def __init__(self, stemmer, stop_words, *args):
        """
        class TextProcessing is used to tokenize and normalize tokens that will be further used to build inverted index.
        :param stemmer:
        :param stop_words:
        :param args:
        """
        self.stemmer = stemmer
        self.STOP_WORDS = stop_words

    @classmethod
    def from_nltk(
        cls,
        stemmer: Any = PorterStemmer().stem,
        stop_words: List[str] = stopwords.words("english"),
    ) -> "TextProcessing":
        """
        initialize from nltk
        :param stemmer:
        :param stop_words:
        :return:
        """
        return cls(stemmer, set(stop_words))

    def normalize(self, token: str) -> str:
        """
        normalize the token based on: 1. make all characters in the token to lower case 2. remove any characters from
        the token other than alphanumeric characters and dash ("-") 3. after step 1, if the processed token appears
        in the stop words list or its length is 1, return an empty string 4. after step 1, if the processed token is
        NOT in the stop words list and its length is greater than 1, return the stem of the token
        :param token:
        :return:
        """
        lower = token.lower()
        al_num = ""
        for letter in lower:
            if letter.isalnum() or letter == '-':
                al_num += letter
        if (al_num in self.STOP_WORDS) or (len(al_num) == 1):
            return ""
        else:
            return self.stemmer(al_num)

    def get_normalized_tokens(self, title: str, content: str) -> List[str]:
        """
        pass in the title and content_str of each document, and return a list of normalized tokens (exclude the empty
        string) you may want to apply word_tokenize first to get un-normalized tokens first. Note that you don't want
        to remove duplicate tokens as what you did in HW3, because it will later be used to compute term frequency
        :param title:
        :param content:
        :return:
        """
        if title:
            normalized = list(map(self.normalize, word_tokenize(title + " " + content)))
        else:
            normalized = list(map(self.normalize, word_tokenize(content)))
        return list(filter(None, normalized))

    @staticmethod
    def idf(n: int, df: int) -> float:
        """
        compute the logarithmic (base 2) idf score
        :param n: document count N
        :param df: document frequency
        :return:
        """
        return math.log2(n/df)

    @staticmethod
    def tf(freq: int) -> float:
        """
        compute the logarithmic tf (base 2) score
        :param freq: raw term frequency
        :return:
        """
        if freq > 0:
            return 1+math.log2(freq)
        else:
            return 0.0


if __name__ == "__main__":
    pass

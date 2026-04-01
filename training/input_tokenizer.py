import string

import spacy
from spacy.lang.en.stop_words import STOP_WORDS


# Negation words that must be preserved even though spaCy marks them as stopwords.
# Removing these destroys sentiment signal — "not good" becomes "good".
NEGATION_WORDS = frozenset({
    "not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere",
    "n't", "nt",                          # contraction suffixes spaCy splits off
    "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "cannot",
    "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
    "shouldn't", "couldn't", "mustn't",
    "hardly", "barely", "scarcely",       # near-negations that flip meaning
})


class SpacyTextPreprocessor:
    """Tokenizes, lemmatizes, and removes stopwords/punctuation from text using spaCy.

    Negation words are explicitly preserved because stripping them destroys
    sentiment signal (e.g. 'not good' → 'good' would flip the meaning).
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self._nlp = spacy.load(model)
        # Build the effective stopword set by removing negation words from it.
        # This means the main filter loop needs no special-case branching.
        self._stopwords = frozenset(STOP_WORDS) - NEGATION_WORDS
        self._punctuation = frozenset(string.punctuation)

    def tokenize(self, text: str) -> list[str]:
        """Return cleaned, lemmatized tokens with stopwords and punctuation removed.

        Negation words (e.g. 'not', 'never', "n't") are kept as-is (lowercased)
        rather than lemmatized, because lemmatization can alter their form in ways
        that break downstream matching.
        """
        doc = self._nlp(text)
        tokens = []

        for token in doc:
            if token.is_space:
                continue

            raw = token.text.lower().strip()

            # Preserve negation words exactly — do not lemmatize them.
            if raw in NEGATION_WORDS:
                tokens.append(raw)
                continue

            lemma = token.lemma_.lower().strip()

            # Drop stopwords and punctuation.
            if lemma in self._stopwords or lemma in self._punctuation:
                continue

            tokens.append(lemma)

        return tokens
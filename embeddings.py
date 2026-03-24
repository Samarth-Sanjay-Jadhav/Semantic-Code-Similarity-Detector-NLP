"""
embeddings.py
-------------
Provides two embedding backends:

  Word2VecEmbedder  — Trains a gensim Word2Vec model on the submitted
                      code tokens (augmented with a built-in Python
                      code corpus so the vocabulary is richer).

  GloVeEmbedder     — Downloads and caches a pre-trained GloVe model
                      via gensim-data (~66 MB on first use).

Both expose .get_vector(tokens) → np.ndarray (mean-pooled document vec).
"""

import logging
import numpy as np
from typing import List
from gensim.models import Word2Vec
import gensim.downloader as api

logger = logging.getLogger(__name__)

# ─── Built-in Python corpus to bootstrap Word2Vec vocabulary ────────────────
# Each inner list represents one "sentence" (a code pattern).
# This ensures common tokens like KW_for, OP_ASSIGN, LPAREN etc.
# already have context before training on user code.
_BOOTSTRAP_CORPUS: List[List[str]] = [
    ['KW_def', 'var_0', 'LPAREN', 'var_1', 'RPAREN', 'COLON', 'KW_return', 'var_2'],
    ['KW_for', 'var_0', 'KW_in', 'BUILTIN_range', 'LPAREN', 'NUM', 'RPAREN', 'COLON'],
    ['KW_while', 'var_0', 'OP_GT', 'NUM', 'COLON', 'var_0', 'OP_ISUB', 'NUM'],
    ['KW_if', 'var_0', 'OP_EQ', 'var_1', 'COLON', 'KW_else', 'COLON'],
    ['KW_if', 'var_0', 'OP_LT', 'var_1', 'COLON', 'KW_elif', 'var_0', 'OP_GT', 'var_1'],
    ['var_0', 'OP_ASSIGN', 'BUILTIN_int', 'LPAREN', 'BUILTIN_input', 'LPAREN', 'RPAREN', 'RPAREN'],
    ['KW_class', 'var_0', 'COLON', 'KW_def', '__init__', 'LPAREN', 'KW_self', 'RPAREN', 'COLON'],
    ['BUILTIN_print', 'LPAREN', 'STRING_LITERAL', 'RPAREN'],
    ['KW_import', 'var_0', 'KW_from', 'var_1', 'KW_import', 'var_2'],
    ['KW_try', 'COLON', 'KW_except', 'Exception', 'KW_as', 'var_0', 'COLON'],
    ['var_0', 'OP_ASSIGN', 'LBRACKET', 'RBRACKET', 'DOT', 'BUILTIN_append', 'LPAREN', 'var_1', 'RPAREN'],
    ['KW_return', 'BUILTIN_sorted', 'LPAREN', 'var_0', 'RPAREN'],
    ['var_0', 'OP_ASSIGN', 'LBRACKET', 'var_1', 'KW_for', 'var_2', 'KW_in', 'var_3', 'RBRACKET'],
    ['KW_with', 'BUILTIN_open', 'LPAREN', 'var_0', 'RPAREN', 'KW_as', 'var_1', 'COLON'],
    ['OP_ADD', 'OP_SUB', 'OP_MUL', 'OP_DIV', 'OP_MOD', 'OP_POW', 'OP_FLOORDIV'],
    ['OP_EQ', 'OP_NEQ', 'OP_LT', 'OP_GT', 'OP_LTE', 'OP_GTE'],
    ['LBRACE', 'var_0', 'COLON', 'var_1', 'RBRACE'],
    ['KW_def', 'var_0', 'LPAREN', 'var_1', 'RPAREN', 'COLON',
     'KW_if', 'var_1', 'OP_LTE', 'NUM', 'COLON', 'KW_return', 'NUM',
     'KW_return', 'var_0', 'LPAREN', 'var_1', 'OP_SUB', 'NUM', 'RPAREN'],
    ['KW_lambda', 'var_0', 'COLON', 'var_0', 'OP_MUL', 'NUM'],
    ['BUILTIN_len', 'LPAREN', 'var_0', 'RPAREN', 'OP_EQ', 'NUM'],
    ['var_0', 'LBRACKET', 'var_1', 'RBRACKET', 'OP_ASSIGN', 'var_2'],
    ['KW_assert', 'var_0', 'OP_EQ', 'var_1', 'COMMA', 'STRING_LITERAL'],
    ['KW_yield', 'var_0'],
    ['KW_raise', 'Exception', 'LPAREN', 'STRING_LITERAL', 'RPAREN'],
    ['var_0', 'OP_ASSIGN', 'BUILTIN_map', 'LPAREN', 'KW_lambda', 'var_1', 'COLON',
     'var_1', 'OP_MUL', 'NUM', 'COMMA', 'var_2', 'RPAREN'],
    ['KW_not', 'var_0', 'KW_in', 'var_1'],
    ['KW_if', '__name__', 'OP_EQ', 'STRING_LITERAL', 'COLON'],
    ['var_0', 'LPAREN', 'RPAREN', 'DOT', 'BUILTIN_split', 'LPAREN', 'STRING_LITERAL', 'RPAREN'],
    ['KW_global', 'var_0', 'KW_nonlocal', 'var_1'],
    ['KW_pass', 'KW_break', 'KW_continue'],
]


class Word2VecEmbedder:
    """
    Trains a gensim Word2Vec model on the submitted code plus the
    built-in bootstrap corpus for richer semantic context.

    Parameters
    ----------
    vector_size : int   Dimensionality of the embedding space.
    window      : int   Context window size.
    min_count   : int   Minimum token frequency to include.
    epochs      : int   Training iterations.
    """

    def __init__(self, vector_size: int = 100, window: int = 5,
                 min_count: int = 1, epochs: int = 200):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model: Word2Vec | None = None
        self.is_trained = False

    def train(self, token_sequences: List[List[str]]) -> None:
        """
        Train Word2Vec on user-supplied token sequences augmented
        with the built-in bootstrap corpus.
        """
        # Augment: repeat user sequences several times for better convergence
        augmented_user = token_sequences * 10
        corpus = _BOOTSTRAP_CORPUS + augmented_user

        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=2,
            epochs=self.epochs,
            sg=1,        # Skip-gram — better for smaller corpora
            seed=42,
        )
        self.is_trained = True

    def get_vector(self, tokens: List[str]) -> np.ndarray:
        """Mean-pool token vectors into a single document vector."""
        if not self.is_trained:
            raise RuntimeError("Call train() before get_vector().")
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)

    def get_token_coverage(self, tokens: List[str]) -> float:
        """Fraction of tokens present in the trained vocabulary."""
        if not self.is_trained or not tokens:
            return 0.0
        return sum(1 for t in tokens if t in self.model.wv) / len(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.model.wv) if self.model else 0


class GloVeEmbedder:
    """
    Pre-trained GloVe embeddings via gensim-data.

    Downloads ~66 MB on first use and caches locally.
    Tokens are lower-cased and prefixes (KW_, BUILTIN_, OP_) are
    stripped before lookup so more tokens match the GloVe vocabulary.

    Parameters
    ----------
    model_name : str  gensim-data key for the desired GloVe variant.
    """

    AVAILABLE = {
        'glove-wiki-gigaword-50':  50,
        'glove-wiki-gigaword-100': 100,
        'glove-twitter-25':        25,
    }

    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        if model_name not in self.AVAILABLE:
            raise ValueError(f"Unknown model. Choose from: {list(self.AVAILABLE)}")
        self.model_name = model_name
        self.vector_size = self.AVAILABLE[model_name]
        self.model = None
        self.is_loaded = False

    def load(self) -> bool:
        """Download (if needed) and load the GloVe model."""
        try:
            self.model = api.load(self.model_name)
            self.vector_size = self.model.vector_size
            self.is_loaded = True
            return True
        except Exception as exc:
            logger.error("GloVe load failed: %s", exc)
            return False

    def _normalize_token(self, token: str) -> str:
        """Strip internal prefixes and lower-case for GloVe lookup."""
        t = token.lower()
        for prefix in ('kw_', 'builtin_', 'op_', 'lp', 'rp'):
            if t.startswith(prefix):
                t = t[len(prefix):]
                break
        return t

    def get_vector(self, tokens: List[str]) -> np.ndarray:
        """Mean-pool token vectors into a single document vector."""
        if not self.is_loaded:
            raise RuntimeError("Call load() before get_vector().")
        vecs = []
        for token in tokens:
            norm = self._normalize_token(token)
            if norm in self.model:
                vecs.append(self.model[norm])
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)

    def get_token_coverage(self, tokens: List[str]) -> float:
        """Fraction of tokens that map to a GloVe vector."""
        if not self.is_loaded or not tokens:
            return 0.0
        return sum(1 for t in tokens
                   if self._normalize_token(t) in self.model) / len(tokens)

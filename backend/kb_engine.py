"""
kb_engine.py — Knowledge Base search engine for SwagatAI

Search strategy (no heavy ML deps required, works offline):
  1. Exact / near-exact match on normalised question string  → score 1.0
  2. Token overlap (Jaccard-like)                            → score 0.4–0.8
  3. Character n-gram overlap (handles Odia/Hindi scripts)   → score 0.2–0.6
  4. Keyword tag match                                       → score bonus +0.1

The engine rebuilds a simple in-memory index from SQLAlchemy rows and
performs sub-millisecond lookups — no internet, no GPU required.
"""

import re
import json
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger('swagatai.kb_engine')

# Minimum confidence to accept a KB match
MATCH_THRESHOLD = 0.32


class KBEngine:

    def __init__(self):
        self._index: list[dict] = []   # list of prepared entry dicts

    # ── Public API ────────────────────────────────────────────────────────

    def rebuild(self, entries) -> None:
        """Rebuild the in-memory index from a list of KBEntry ORM objects."""
        self._index = [self._prepare(e) for e in entries]
        logger.info(f"KB index rebuilt: {len(self._index)} entries")

    def search(self, question: str, lang: str = 'en') -> Optional[dict]:
        """
        Return the best-matching KB entry dict, or None if nothing passes
        the confidence threshold.

        Returns: {id, answer, score} or None
        """
        if not self._index:
            return None

        q_norm   = self._normalise(question)
        q_tokens = self._tokenise(q_norm)
        q_ngrams = self._ngrams(q_norm, n=3)

        best_score = 0.0
        best_entry = None

        for entry in self._index:
            score = self._score(q_norm, q_tokens, q_ngrams, entry, lang)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= MATCH_THRESHOLD:
            logger.info(f"KB hit score={best_score:.3f} id={best_entry['id']}")
            return {
                'id':     best_entry['id'],
                'answer': best_entry['answer'],
                'score':  best_score,
            }

        logger.info(f"KB miss best_score={best_score:.3f} q={question!r}")
        return None

    # ── Private ───────────────────────────────────────────────────────────

    def _prepare(self, entry) -> dict:
        q_norm = self._normalise(entry.question)
        return {
            'id':       entry.id,
            'question': entry.question,
            'q_norm':   q_norm,
            'q_tokens': set(self._tokenise(q_norm)),
            'q_ngrams': set(self._ngrams(q_norm, n=3)),
            'answer':   entry.answer,
            'lang':     entry.lang,
            'tags':     set(json.loads(entry.tags or '[]')),
        }

    def _score(self, q_norm: str, q_tokens: list, q_ngrams: set, entry: dict, lang: str) -> float:
        score = 0.0

        # 1. Exact normalised match
        if q_norm == entry['q_norm']:
            return 1.0

        # 2. One is substring of the other
        if q_norm in entry['q_norm'] or entry['q_norm'] in q_norm:
            score = max(score, 0.75)

        # 3. Token Jaccard similarity
        e_tokens = entry['q_tokens']
        qt_set   = set(q_tokens)
        if qt_set and e_tokens:
            intersection = len(qt_set & e_tokens)
            union        = len(qt_set | e_tokens)
            jaccard      = intersection / union if union else 0
            score        = max(score, jaccard * 0.85)

            # Bonus: all query tokens found in entry
            if qt_set and qt_set.issubset(e_tokens):
                score = max(score, 0.72)

        # 4. Character n-gram overlap (good for Odia/Hindi)
        if q_ngrams and entry['q_ngrams']:
            ng_inter = len(q_ngrams & entry['q_ngrams'])
            ng_union = len(q_ngrams | entry['q_ngrams'])
            ng_score = ng_inter / ng_union if ng_union else 0
            score    = max(score, ng_score * 0.7)

        # 5. Tag bonus
        for token in q_tokens:
            if token in entry['tags']:
                score += 0.1
                break

        # 6. Language affinity bonus (same lang = slight boost)
        if lang == entry['lang']:
            score *= 1.05

        return min(score, 1.0)

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, strip punctuation, Unicode NFKC normalise."""
        text = unicodedata.normalize('NFKC', text)
        text = text.lower()
        # Remove common punctuation but keep Devanagari/Odia characters
        text = re.sub(r'[?।!,;:"\'\(\)\[\]{}]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _tokenise(text: str) -> list:
        """Split on whitespace and filter very short tokens."""
        return [t for t in text.split() if len(t) > 1]

    @staticmethod
    def _ngrams(text: str, n: int = 3) -> set:
        """Character n-grams (compact sliding window)."""
        text = text.replace(' ', '_')
        return {text[i:i+n] for i in range(len(text) - n + 1)}

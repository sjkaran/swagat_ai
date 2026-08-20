"""
ai_fallback.py — Lightweight AI fallback for SwagatAI

Priority order:
  1. Ollama (local LLM — runs on-device, no internet needed)
  2. Rule-based polite stub (works 100% offline, zero dependencies)

The stub provides dignified multilingual responses so visitors are never
left with an error message, even on the cheapest hardware with no internet.
"""

import logging
import random
from typing import Any

logger = logging.getLogger('swagatai.ai_fallback')

# ─── MULTILINGUAL STUBS ────────────────────────────────────────────────────

STUBS: dict[str, list[str]] = {
    'en': [
        "Thank you for your question. I don't have that specific information right now, "
        "but our staff at the reception desk will be happy to assist you.",
        "That's a great question! I'm still learning. Please check with our information "
        "desk for accurate details — they're always ready to help.",
        "I appreciate you asking. I don't have that answer in my knowledge yet, "
        "but I've noted your question so our team can train me soon.",
        "I'm sorry, I couldn't find that information. Please speak to our staff member "
        "on duty, or call our helpline number displayed at the entrance.",
    ],
    'hi': [
        "आपके प्रश्न के लिए धन्यवाद। मेरे पास अभी यह जानकारी नहीं है, "
        "लेकिन हमारे रिसेप्शन स्टाफ आपकी मदद करने के लिए तैयार हैं।",
        "यह एक अच्छा प्रश्न है! मैं अभी सीख रही हूँ। कृपया हमारी जानकारी "
        "काउंटर से संपर्क करें — वे आपकी सहायता करेंगे।",
        "मुझे खेद है, मेरे पास अभी यह उत्तर नहीं है। कृपया ड्यूटी पर मौजूद "
        "हमारे स्टाफ से बात करें।",
    ],
    'or': [
        "ଆପଣଙ୍କ ପ୍ରଶ୍ନ ପାଇଁ ଧନ୍ୟବାଦ। ଏହି ସୂଚନା ବର୍ତ୍ତମାନ ମୋ ପାଖରେ ନାହିଁ, "
        "କିନ୍ତୁ ଆମ ରିସେପ୍ସନ ସ୍ଟାଫ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବେ।",
        "ଭଲ ପ୍ରଶ୍ନ! ମୁଁ ଏଯାଏ ଏହା ଶିଖି ନାହିଁ। ଦୟାକରି ଆମ ତଥ୍ୟ ଡେସ୍କ ପାଖକୁ ଯାଆନ୍ତୁ।",
        "ଦୁଃଖିତ, ଏ ଉତ୍ତର ବର୍ତ୍ତମାନ ମୋ ଜ୍ଞାନରେ ନାହିଁ। ଦ୍ୱାର ଉପରେ ଲେଖା ଆମ "
        "ହେଲ୍ପଲାଇନ ନମ୍ବରରେ ଯୋଗାଯୋଗ କରନ୍ତୁ।",
    ],
}

# System prompt for Ollama (concise, multilingual-aware)
SYSTEM_PROMPT = """You are Swagata, a helpful AI reception assistant for a business or
public institution in Odisha, India. You speak Odia (or), Hindi (hi), and English (en).
Answer visitor questions briefly, politely, and accurately in the same language as the
question. If you don't know, admit it gracefully and suggest asking staff. Keep responses
under 60 words. No markdown."""


class AIFallback:

    def __init__(self, model: str = 'llama3.2:3b', base_url: str = 'http://localhost:11434'):
        self.model    = model
        self.base_url = base_url.rstrip('/')
        self._ollama_ok: bool | None = None  # None = untested

    # ── Public API ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Quick check whether Ollama is reachable."""
        if self._ollama_ok is None:
            self._ollama_ok = self._ping_ollama()
        return self._ollama_ok

    def answer(self, question: str, lang: str = 'en') -> dict[str, Any]:
        """
        Return {'answer': str, 'confidence': float, 'source': str}
        Tries Ollama first; falls back to polite stub.
        """
        if self.is_available():
            result = self._ask_ollama(question, lang)
            if result:
                return result

        # Stub fallback (always works)
        return {
            'answer':     self._stub(lang),
            'confidence': 0.3,
            'source':     'stub',
        }

    # ── Ollama ──────────────────────────────────────────────────────────

    def _ping_ollama(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(f'{self.base_url}/api/tags', timeout=2)
            logger.info("Ollama is reachable.")
            return True
        except Exception as e:
            logger.info(f"Ollama not available ({e}); will use stub.")
            return False

    def _ask_ollama(self, question: str, lang: str) -> dict | None:
        try:
            import json as _json
            import urllib.request, urllib.error

            payload = _json.dumps({
                'model':  self.model,
                'stream': False,
                'system': SYSTEM_PROMPT,
                'prompt': f"[Language: {lang}] Visitor question: {question}",
                'options': {'temperature': 0.4, 'num_predict': 120},
            }).encode()

            req = urllib.request.Request(
                f'{self.base_url}/api/generate',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = _json.loads(resp.read())
                answer = body.get('response', '').strip()
                if answer:
                    return {'answer': answer, 'confidence': 0.65, 'source': 'ollama'}
        except Exception as e:
            logger.warning(f"Ollama error: {e}")
            self._ollama_ok = False  # stop retrying this session
        return None

    # ── Stub ────────────────────────────────────────────────────────────

    def _stub(self, lang: str) -> str:
        pool = STUBS.get(lang, STUBS['en'])
        return random.choice(pool)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import os
import io
import re
import json
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor, VitsModel, AutoTokenizer
import sounddevice as sd
import threading
import time
import queue
import scipy.io.wavfile
import scipy.signal

# ==========================================
# FUZZY MATCHING & LANGUAGE DETECTION IMPORTS
# ==========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42  # Make langdetect deterministic

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "swagat_ai.db")

# ==========================================
# 0. INITIALIZE ODIA AI ENGINES
# ==========================================
print("Loading Odia AI Engines... This may take a moment.")
try:
    with open(os.path.join(BASE_DIR, "source.txt"), "r") as key:
        HF_TOKEN = key.readline().strip()
except FileNotFoundError:
    print("WARNING: source.txt not found. Models may fail to download.")
    HF_TOKEN = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load STT
stt_proc = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec-odia", token=HF_TOKEN)
stt_model = AutoModelForCTC.from_pretrained("ai4bharat/indicwav2vec-odia", token=HF_TOKEN).to(DEVICE)

# Load TTS
tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ory", token=HF_TOKEN)
tts_model = VitsModel.from_pretrained("facebook/mms-tts-ory", token=HF_TOKEN).to(DEVICE)
print(f"Odia Engines Ready on {DEVICE}!")

# ==========================================
# 1. DATABASE INITIALIZATION
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        language TEXT NOT NULL,
        ai_cached INTEGER DEFAULT 0
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS business_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        biz_type TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

def migrate_db():
    """Safely add ai_cached column if upgrading from an older DB."""
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute('ALTER TABLE knowledge_base ADD COLUMN ai_cached INTEGER DEFAULT 0')
        conn.commit()
        print("[DB] Migration complete: added ai_cached column.")
    except sqlite3.OperationalError:
        pass  # Column already exists — nothing to do
    finally:
        conn.close()

if not os.path.exists(DB_NAME):
    init_db()
migrate_db()  # Safe to run every startup — no-op if already migrated

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# ==========================================
# 2. ENHANCED AUDIO PRE-PROCESSING
# ==========================================
def preprocess_audio(audio_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Apply noise-reduction pipeline before passing to wav2vec:
    1. Convert stereo → mono (safety)
    2. Highpass filter at 80Hz to kill low-frequency rumble
    3. Spectral subtraction for background noise removal
    4. Voice Activity Detection (VAD) trim to drop silent edges
    5. Normalize amplitude
    """
    # Step 1: Ensure mono float32
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)

    # Step 2: Highpass filter (80 Hz cutoff) — removes rumble & mic pop
    b, a = scipy.signal.butter(4, 80 / (sr / 2), btype='high')
    audio_np = scipy.signal.filtfilt(b, a, audio_np)

    # Step 3: Spectral subtraction (simple but effective noise gate)
    # Estimate noise profile from first 200ms of audio (assumed silence/pre-speech)
    noise_samples = min(int(0.2 * sr), len(audio_np) // 4)
    if noise_samples > 0:
        noise_profile = np.abs(np.fft.rfft(audio_np[:noise_samples]))
        n_fft = len(audio_np)
        spectrum = np.fft.rfft(audio_np, n=n_fft)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        # Pad noise profile to match spectrum length
        noise_pad = np.pad(noise_profile, (0, len(magnitude) - len(noise_profile)), 'edge')
        # Subtract noise floor with over-subtraction factor α=2.0
        alpha = 2.0
        cleaned_magnitude = np.maximum(magnitude - alpha * noise_pad, 0.01 * magnitude)
        audio_np = np.fft.irfft(cleaned_magnitude * np.exp(1j * phase), n=n_fft)
        audio_np = audio_np[:len(audio_np)]

    # Step 4: VAD trim — remove leading/trailing silence
    # Frame-based energy threshold
    frame_len = int(0.02 * sr)  # 20ms frames
    hop = frame_len // 2
    energies = np.array([
        np.mean(audio_np[i:i+frame_len] ** 2)
        for i in range(0, len(audio_np) - frame_len, hop)
    ])
    if len(energies) > 0:
        threshold = np.percentile(energies, 15) * 6  # 6x noise floor
        voiced = np.where(energies > threshold)[0]
        if len(voiced) > 0:
            start = max(0, voiced[0] * hop - frame_len)
            end = min(len(audio_np), voiced[-1] * hop + frame_len * 3)
            audio_np = audio_np[start:end]

    # Step 5: Normalize to [-1, 1]
    max_val = np.abs(audio_np).max()
    if max_val > 1e-6:
        audio_np = audio_np / max_val * 0.95

    return audio_np

# ==========================================
# 3. AUTO LANGUAGE DETECTION
# ==========================================

# Odia Unicode block: U+0B00–U+0B7F
ODIA_UNICODE_PATTERN = re.compile(r'[\u0B00-\u0B7F]')
# Devanagari block (Hindi): U+0900–U+097F
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')

def detect_language(text: str) -> str:
    """
    Detect language from transcribed text.
    Returns ISO code: 'or' (Odia), 'hi' (Hindi), 'en' (English)
    Priority: Script-based check first (fast & reliable) → langdetect fallback
    """
    if not text or len(text.strip()) < 2:
        return 'en'  # Default

    # Fast script-based detection (most reliable for Indian scripts)
    odia_chars = len(ODIA_UNICODE_PATTERN.findall(text))
    hindi_chars = len(DEVANAGARI_PATTERN.findall(text))
    total_chars = len(text.replace(' ', ''))

    if total_chars > 0:
        if odia_chars / total_chars > 0.3:
            return 'or'
        if hindi_chars / total_chars > 0.3:
            return 'hi'

    # Fallback: langdetect for Roman script (English detection)
    try:
        detected = detect(text)
        if detected in ('or', 'hi', 'en'):
            return detected
        # langdetect may return 'bn' for some Odia text — remap
        if detected == 'bn' and odia_chars > 0:
            return 'or'
        if detected in ('mr', 'ne') and hindi_chars > 0:
            return 'hi'
    except Exception:
        pass

    return 'en'

# ==========================================
# 4. MULTI-SIGNAL QUERY MATCHING ENGINE
# ==========================================

# Threshold: a combined score below this goes to AI fallback
SIMILARITY_THRESHOLD = 0.52

# Common question words to strip before comparison (English + Odia + Hindi)
STOPWORDS = {
    'what', 'where', 'when', 'how', 'which', 'who', 'is', 'are', 'the',
    'a', 'an', 'do', 'does', 'can', 'i', 'me', 'my', 'tell', 'please',
    'your', 'you', 'of', 'for', 'in', 'to', 'and', 'or', 'at', 'this',
    'that', 'get', 'find', 'know', 'much', 'many', 'there', 'any',
    # Hindi stopwords
    'क्या', 'कहाँ', 'कब', 'कैसे', 'कौन', 'है', 'हैं', 'मैं', 'मुझे',
    'आप', 'का', 'की', 'के', 'से', 'में', 'को', 'और', 'या', 'बताइए',
    'कितना', 'कितनी', 'मिलेगा', 'मिलती', 'होता', 'होती',
    # Odia stopwords
    'କଣ', 'କେଉଁ', 'କେତେ', 'କିପରି', 'ଆପଣ', 'ମୁଁ', 'ଏହା', 'ସେ',
    'ଆଉ', 'ବା', 'ରେ', 'ର', 'ଟି', 'ଟା',
}

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    # Keep Odia (U+0B00-U+0B7F), Devanagari (U+0900-U+097F), word chars, spaces
    text = re.sub(r'[^\w\s\u0B00-\u0B7F\u0900-\u097F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_stopwords(text: str) -> str:
    """Strip common filler/question words so content words are compared."""
    tokens = normalize_text(text).split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(filtered) if filtered else normalize_text(text)

def keyword_overlap_score(query: str, candidate: str) -> float:
    """
    Jaccard overlap on content keywords (stopwords removed).
    Handles cases like 'ticket price' vs 'How much does a ticket cost?' — 
    shares 'ticket' → good signal.
    """
    q_tokens = set(remove_stopwords(query).split())
    c_tokens = set(remove_stopwords(candidate).split())
    if not q_tokens or not c_tokens:
        return 0.0
    intersection = q_tokens & c_tokens
    # Use smaller set as denominator — rewards partial coverage
    smaller = min(len(q_tokens), len(c_tokens))
    return len(intersection) / smaller if smaller > 0 else 0.0

def tfidf_cosine_score(query: str, questions: list) -> list:
    """
    TF-IDF char n-gram cosine similarity.
    Returns a list of float scores aligned with `questions`.
    Falls back to zeros on error (e.g. single-item KB).
    """
    if not questions:
        return []
    try:
        corpus = [query] + questions
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(matrix[0], matrix[1:]).flatten()
        return [float(s) for s in scores]
    except Exception as e:
        print(f"[TF-IDF] Error: {e}")
        return [0.0] * len(questions)

def is_indic(text: str) -> bool:
    """Returns True if text contains significant Odia or Devanagari characters."""
    indic_chars = len(re.findall(r'[\u0900-\u097F\u0B00-\u0B7F]', text))
    return indic_chars > 1

def score_entry(query_norm: str, query_kw: str, candidate_norm: str, candidate_kw: str) -> float:
    """
    Compute a single combined score for one (query, candidate) pair using
    4 independent signals. Each is normalised to [0, 1].

    For Indic scripts (Odia/Hindi), partial_ratio weight is boosted because:
    - Indic words share root characters even when synonyms differ
    - Word boundary splitting is less reliable for agglutinative Indic morphology

    Signal breakdown (Latin):
      1. token_set_ratio   — reordered / paraphrased queries    (weight 0.35)
      2. partial_ratio     — substring / keyword presence        (weight 0.25)
      3. keyword_overlap   — content-word matching               (weight 0.25)
      4. token_sort_ratio  — word-order invariant ratio          (weight 0.15)

    Signal breakdown (Indic):
      1. partial_ratio     — char-level substring overlap        (weight 0.45)
      2. token_set_ratio   — reordering                         (weight 0.30)
      3. keyword_overlap   — root word matching                  (weight 0.15)
      4. token_sort_ratio  — order invariant                     (weight 0.10)
    """
    s1 = fuzz.token_set_ratio(query_norm, candidate_norm) / 100.0
    s2 = fuzz.partial_ratio(query_norm, candidate_norm) / 100.0
    s3 = keyword_overlap_score(query_kw, candidate_kw)
    s4 = fuzz.token_sort_ratio(query_norm, candidate_norm) / 100.0

    if is_indic(query_norm) or is_indic(candidate_norm):
        # Boost partial_ratio for Indic scripts
        return 0.45 * s2 + 0.30 * s1 + 0.15 * s3 + 0.10 * s4
    else:
        return 0.35 * s1 + 0.25 * s2 + 0.25 * s3 + 0.15 * s4

def find_best_kb_match(user_query: str, kb_items: list, language_filter: str = None) -> dict | None:
    """
    Full matching pipeline:

    Pass 1 — Exact / substring match (instant, score = 1.0)
    Pass 2 — Multi-signal fuzzy scoring on language-filtered items
    Pass 3 — If nothing clears threshold, retry across all languages
    Pass 4 — TF-IDF re-rank the top-5 candidates as a final tiebreaker

    Returns a result dict: { item, score, match_type, signals } or None.
    """
    if not kb_items:
        return None

    # Decide search pool: prefer matching language, fall back to all
    if language_filter:
        lang_items = [item for item in kb_items if item['language'] == language_filter]
        search_pool = lang_items if lang_items else list(kb_items)
    else:
        search_pool = list(kb_items)

    query_norm = normalize_text(user_query)
    query_kw   = remove_stopwords(user_query)

    # ── PASS 1: Exact / substring match ──────────────────────────────────────
    for item in search_pool:
        cand_norm = normalize_text(item['question'])
        if cand_norm == query_norm:
            return _result(item, 1.0, 'exact', {})
        if cand_norm in query_norm or query_norm in cand_norm:
            return _result(item, 0.97, 'substring', {})

    # ── PASS 2: Per-candidate multi-signal scoring ────────────────────────────
    scored = []
    for item in search_pool:
        cand_norm = normalize_text(item['question'])
        cand_kw   = remove_stopwords(item['question'])
        combined  = score_entry(query_norm, query_kw, cand_norm, cand_kw)
        signals   = {
            'token_set': round(fuzz.token_set_ratio(query_norm, cand_norm) / 100, 3),
            'partial':   round(fuzz.partial_ratio(query_norm, cand_norm) / 100, 3),
            'keyword':   round(keyword_overlap_score(query_kw, cand_kw), 3),
            'token_sort':round(fuzz.token_sort_ratio(query_norm, cand_norm) / 100, 3),
        }
        scored.append((combined, item, signals))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_item, best_signals = scored[0]

    # ── PASS 3: Cross-language retry if nothing good enough ───────────────────
    if best_score < SIMILARITY_THRESHOLD and language_filter:
        cross_pool = [i for i in kb_items if i['language'] != language_filter]
        for item in cross_pool:
            cand_norm = normalize_text(item['question'])
            cand_kw   = remove_stopwords(item['question'])
            combined  = score_entry(query_norm, query_kw, cand_norm, cand_kw)
            if combined > best_score:
                best_score = combined
                best_item  = item
                best_signals = {'cross_lang': True}

    if best_score < SIMILARITY_THRESHOLD:
        print(f"[Matcher] No match for '{user_query[:40]}' (best={best_score:.2f})")
        return None

    # ── PASS 4: TF-IDF re-rank the top-5 candidates ──────────────────────────
    top5 = scored[:5]
    top5_questions = [normalize_text(i['question']) for _, i, _ in top5]
    tfidf_scores = tfidf_cosine_score(query_norm, top5_questions)

    # Blend TF-IDF boost (15%) into top-5 scores to break ties
    reranked = [
        (score + 0.15 * tf, item, sigs)
        for (score, item, sigs), tf in zip(top5, tfidf_scores)
    ]
    reranked.sort(key=lambda x: x[0], reverse=True)
    final_score, final_item, final_sigs = reranked[0]
    final_score = min(final_score, 1.0)  # Cap at 100%

    match_type = 'exact' if final_score >= 0.97 else 'fuzzy'
    print(f"[Matcher] '{user_query[:40]}' → '{final_item['question'][:40]}' score={final_score:.2f}")
    return _result(final_item, round(final_score, 3), match_type, final_sigs)

def _result(item, score, match_type, signals):
    return {
        'item': dict(item),
        'score': score,
        'match_type': match_type,
        'signals': signals
    }

# ==========================================
# ==========================================
# 5. AI FALLBACK -- Ollama (local, fully offline, no API key needed)
# ==========================================
#
# HOW TO SET UP (one-time, ~5 minutes):
#   1. Download Ollama from: https://ollama.com/download
#   2. Install it, then open a NEW terminal and run:
#        ollama pull mistral        <- 4.1 GB, best quality
#      OR smaller:
#        ollama pull llama3.2:3b   <- 2.0 GB, faster on weaker machines
#   3. Ollama runs as a background service on port 11434 automatically.
#   4. Done -- no token, no internet needed after the initial model download.
#
# To switch models: change OLLAMA_MODEL below to any model you have pulled.
# List what you have: run  ollama list  in terminal.
#

OLLAMA_MODEL    = "mistral"                  # change to e.g. "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"   # Ollama default

import requests as _requests  # aliased to avoid shadowing Flask's `request`

# Edit this prompt to describe your venue/business:
def build_system_prompt(biz_context: dict) -> str:
    return (
        f"You are Swagata, a warm and helpful AI reception assistant for "
        f"{biz_context['name']}, a {biz_context['type']} located in Odisha, India. "
        f"Answer visitor questions about the venue, timings, tickets, and facilities. "
        f"Keep every reply to 2-3 sentences -- this is a reception kiosk. "
        f"Do NOT invent specific details like phone numbers, prices, or dates. "
        f"If you do not know, say so politely and suggest asking the front desk staff. "
        f"If the visitor writes in Odia or Hindi, reply in that same language."
    )


def _clean_answer(text: str) -> str:
    """Strip role-echo prefixes the model sometimes prepends."""
    for prefix in ("Swagata:", "Assistant:", "AI:", "Response:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip()


def _check_ollama_running() -> bool:
    """Quick ping -- is Ollama up on localhost:11434?"""
    try:
        r = _requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def generate_ai_fallback_response(user_query: str, biz_context: dict) -> dict:
    """
    Call Ollama's OpenAI-compatible local endpoint.
    URL: http://localhost:11434/v1/chat/completions
    Fully offline after model download -- no token, no rate limits.
    """
    if not _check_ollama_running():
        print("[AI Fallback] Ollama not detected on port 11434.")
        return {
            "answer": (
                "My AI assistant is currently offline. "
                "Please ensure Ollama is running (run 'ollama serve' in terminal), "
                "then try again or ask our front desk staff."
            ),
            "model": OLLAMA_MODEL,
            "ai_source": "ollama_offline",
        }

    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt(biz_context)},
            {"role": "user",   "content": user_query},
        ],
        "max_tokens": 200,
        "temperature": 0.4,
        "top_p": 0.85,
        "stream": False,
    }

    try:
        print(f"[AI Fallback] Ollama POST {url} | model={OLLAMA_MODEL}")
        resp = _requests.post(url, json=payload, timeout=60)
        print(f"[AI Fallback] HTTP {resp.status_code}")

        if resp.status_code != 200:
            print(f"[AI Fallback] Error body: {resp.text[:400]}")

        if resp.status_code == 404:
            return {
                "answer": (
                    f"Model '{OLLAMA_MODEL}' is not downloaded yet. "
                    f"Please run: ollama pull {OLLAMA_MODEL}"
                ),
                "model": OLLAMA_MODEL,
                "ai_source": "model_not_found",
            }

        resp.raise_for_status()
        data   = resp.json()
        answer = data["choices"][0]["message"]["content"]
        answer = _clean_answer(answer)
        print(f"[AI Fallback] SUCCESS ({len(answer)} chars): {answer[:100]}")
        return {"answer": answer, "model": OLLAMA_MODEL, "ai_source": "ollama_local"}

    except _requests.ConnectionError:
        print("[AI Fallback] Connection refused -- Ollama not running.")
        return {
            "answer": "My local AI is not reachable. Please start Ollama and try again, or ask our staff.",
            "model": OLLAMA_MODEL, "ai_source": "ollama_offline",
        }
    except _requests.Timeout:
        print("[AI Fallback] Ollama timed out.")
        return {
            "answer": "The AI is taking too long. Please try again in a moment.",
            "model": OLLAMA_MODEL, "ai_source": "ollama_timeout",
        }
    except (KeyError, IndexError, TypeError) as e:
        print(f"[AI Fallback] Parse error: {e}")
        return {
            "answer": "I received an unexpected AI response. Please ask our front desk staff.",
            "model": OLLAMA_MODEL, "ai_source": "parse_error",
        }
    except Exception as e:
        print(f"[AI Fallback] Unexpected: {type(e).__name__}: {e}")
        return {
            "answer": "Something went wrong with AI. Please ask our staff directly.",
            "model": OLLAMA_MODEL, "ai_source": "unknown_error",
        }


# 6. ODIA AI ENDPOINTS (Hardware Mic Control)
# ==========================================
odia_mic_state = {"is_recording": False}
audio_queue = queue.Queue()

def record_audio_task():
    """Background thread capturing pure, raw mic audio."""
    def callback(indata, frames, time, status):
        if status:
            print(f"Mic Status: {status}")
        if odia_mic_state["is_recording"]:
            audio_queue.put(indata.copy())

    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=callback):
            while odia_mic_state["is_recording"]:
                sd.sleep(100)
    except Exception as e:
        print(f"Hardware Mic Error: {e}")
        odia_mic_state["is_recording"] = False

@app.route('/api/listen/odia/start', methods=['POST'])
def start_odia_listen():
    if odia_mic_state["is_recording"]:
        return jsonify({"status": "already recording"})

    odia_mic_state["is_recording"] = True
    while not audio_queue.empty():
        audio_queue.get()

    threading.Thread(target=record_audio_task, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/api/listen/odia/stop', methods=['POST'])
def stop_odia_listen():
    odia_mic_state["is_recording"] = False
    time.sleep(0.3)

    frames = []
    while not audio_queue.empty():
        frames.append(audio_queue.get())

    if not frames:
        return jsonify({"text": "", "detected_lang": "or"})

    # Stitch raw chunks
    audio_np = np.concatenate(frames, axis=0)
    audio_np = np.squeeze(audio_np)

    # ENHANCEMENT 1: Apply noise reduction & VAD before sending to model
    audio_np = preprocess_audio(audio_np, sr=16000)

    if len(audio_np) < 800:  # Less than 50ms — too short, skip
        return jsonify({"text": "", "detected_lang": "or"})

    # Pass cleaned audio to wav2vec
    inputs = stt_proc(audio_np, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = stt_model(inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    text = stt_proc.batch_decode(pred_ids, skip_special_tokens=True)[0]

    # ENHANCEMENT 3: Detect actual language of transcribed text
    detected_lang = detect_language(text)

    return jsonify({"text": text, "detected_lang": detected_lang})

# ==========================================
# GENERIC STT ENDPOINT (for Web Speech API results)
# ENHANCEMENT 3: Auto-detect language from any transcription
# ==========================================
@app.route('/api/detect-lang', methods=['POST'])
def detect_lang_endpoint():
    """Given transcribed text, return detected language code."""
    text = request.json.get("text", "")
    detected = detect_language(text)
    return jsonify({"detected_lang": detected})

@app.route('/api/tts/odia', methods=['POST'])
def tts_odia():
    text = request.json.get("text", "").strip()

    if not text:
        wav_io = io.BytesIO()
        scipy.io.wavfile.write(wav_io, rate=16000, data=np.zeros(160, dtype=np.int16))
        wav_io.seek(0)
        return send_file(wav_io, mimetype="audio/wav")

    inputs = tts_tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_audio = tts_model(**inputs).waveform[0].cpu().numpy()

    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, rate=tts_model.config.sampling_rate, data=output_audio)
    wav_io.seek(0)
    return send_file(wav_io, mimetype="audio/wav")

# ==========================================
# 7. STANDARD ENDPOINTS
# ==========================================
@app.route('/')
def serve_frontend():
    return send_file('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_assistant():
    data = request.json
    user_query = data.get('query', '').strip()
    language = data.get('language', 'en')

    conn = get_db_connection()
    kb_items = conn.execute('SELECT * FROM knowledge_base').fetchall()

    # Multi-signal fuzzy match — language-aware first, cross-language fallback built-in
    match = find_best_kb_match(user_query, kb_items, language_filter=language)

    if match:
        conn.close()
        return jsonify({
            "answer": match['item']['answer'],
            "matched_question": match['item']['question'],   # The KB question that matched
            "source": "knowledge_base",
            "match_type": match['match_type'],
            "confidence": match['score'],
            "signals": match.get('signals', {})
        })

    profile = conn.execute('SELECT * FROM business_profile LIMIT 1').fetchone()
    conn.close()
    biz_context = {
        "name": profile['name'] if profile else "Odisha State Museum",
        "type": profile['biz_type'] if profile else "Heritage Site / Museum"
    }

    # ── AI Fallback: Ollama local model ──────────────────────────────────────
    ai_result = generate_ai_fallback_response(user_query, biz_context)
    answer    = ai_result["answer"]
    ai_source = ai_result["ai_source"]

    # Auto-cache the AI answer into KB so repeat queries are instant.
    # Marked as ai_cached=1 so admin can review/refine/delete it separately.
    # Only cache genuine answers — skip error/offline/timeout responses.
    skip_cache = {"ollama_offline", "model_not_found", "ollama_timeout",
                  "parse_error", "unknown_error", "no_token"}
    if ai_source not in skip_cache:
        try:
            cache_conn = get_db_connection()
            cache_conn.execute(
                'INSERT INTO knowledge_base (question, answer, language, ai_cached) VALUES (?, ?, ?, 1)',
                (user_query, answer, language)
            )
            cache_conn.commit()
            cache_conn.close()
            print(f"[KB Cache] Saved AI answer for: '{user_query[:50]}'")
        except Exception as e:
            print(f"[KB Cache] Failed to save: {e}")

    return jsonify({
        "answer":     answer,
        "source":     "ai_fallback",
        "ai_model":   ai_result["model"],
        "ai_source":  ai_source,
        "confidence": 0.0
    })

@app.route('/api/ai-status', methods=['GET'])
def ai_status():
    """Returns current AI fallback configuration status for the frontend."""
    ollama_running = _check_ollama_running()
    return jsonify({
        "model":          OLLAMA_MODEL,
        "token_configured": True,          # Ollama needs no token
        "ollama_running": ollama_running,
        "ollama_url":     OLLAMA_BASE_URL,
    })

@app.route('/api/kb', methods=['GET'])
def get_knowledge_base():
    conn = get_db_connection()
    items = conn.execute('SELECT * FROM knowledge_base ORDER BY id DESC').fetchall()
    conn.close()
    return jsonify([dict(ix) for ix in items])

@app.route('/api/kb', methods=['POST'])
def create_kb_entry():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO knowledge_base (question, answer, language) VALUES (?, ?, ?)',
        (data['question'], data['answer'], data['language'])
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return jsonify({"id": new_id, "question": data['question'], "answer": data['answer'], "language": data['language']})

@app.route('/api/kb/<int:item_id>', methods=['DELETE'])
def delete_kb_entry(item_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM knowledge_base WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Deleted successfully"})

@app.route('/api/kb/<int:item_id>/promote', methods=['POST'])
def promote_kb_entry(item_id):
    """Convert an AI-cached entry into a permanent trained entry (ai_cached=0)."""
    conn = get_db_connection()
    conn.execute('UPDATE knowledge_base SET ai_cached = 0 WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()
    print(f"[KB] Entry {item_id} promoted to trained.")
    return jsonify({"status": "success"})

@app.route('/api/profile', methods=['GET', 'POST'])
def handle_profile():
    conn = get_db_connection()
    if request.method == 'GET':
        profile = conn.execute('SELECT * FROM business_profile LIMIT 1').fetchone()
        conn.close()
        return jsonify(dict(profile)) if profile else jsonify({"name": "Odisha State Museum", "biz_type": "Heritage Site / Museum"})
    else:
        data = request.json
        cursor = conn.cursor()
        profile = cursor.execute('SELECT id FROM business_profile LIMIT 1').fetchone()
        if not profile:
            cursor.execute('INSERT INTO business_profile (name, biz_type) VALUES (?, ?)', (data['name'], data['biz_type']))
        else:
            cursor.execute('UPDATE business_profile SET name = ?, biz_type = ? WHERE id = ?', (data['name'], data['biz_type'], profile['id']))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
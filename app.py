from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import os
import io
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor, VitsModel, AutoTokenizer
import sounddevice as sd
import threading
import time

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "swagat_ai.db")
import scipy.io.wavfile

# ==========================================
# 0. INITIALIZE ODIA AI ENGINES
# ==========================================
print("Loading Odia AI Engines... This may take a moment.")
try:
    with open(os.path.join(BASE_DIR,"source.txt"), "r") as key:
        HF_TOKEN = key.readline().strip()
except FileNotFoundError:
    print("WARNING: API_KEY.txt not found. Models may fail to download.")
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
    cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge_base (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, answer TEXT NOT NULL, language TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS business_profile (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, biz_type TEXT NOT NULL)''')
    conn.commit()
    conn.close()

if not os.path.exists(DB_NAME):
    init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# ==========================================
# 2. AI FALLBACK MODULE
# ==========================================
def generate_ai_fallback_response(user_query, biz_context):
    return "I am currently unable to connect to my AI fallback brain. Please contact the staff directly."

# ==========================================
# 3. ODIA AI ENDPOINTS (Hardware Mic Control)
# ==========================================
# Global state to hold audio chunks while recording
odia_mic_state = {"is_recording": False, "frames": []}

def record_audio_task():
    """Background thread that captures audio chunks directly from the system mic"""
    def callback(indata, frames, time, status):
        if odia_mic_state["is_recording"]:
            odia_mic_state["frames"].append(indata.copy())

    # 16kHz, mono, float32 - perfect format for indicwav2vec
    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=callback):
            while odia_mic_state["is_recording"]:
                sd.sleep(100)
    except Exception as e:
        print(f"Hardware Mic Error: {e}")
        odia_mic_state["is_recording"] = False

@app.route('/api/listen/odia/start', methods=['POST'])
def start_odia_listen():
    """Triggered when user first taps the mic"""
    if odia_mic_state["is_recording"]:
        return jsonify({"status": "already recording"})
    
    odia_mic_state["is_recording"] = True
    odia_mic_state["frames"] = []
    
    # Start the recording loop in the background so the server doesn't freeze
    threading.Thread(target=record_audio_task, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/api/listen/odia/stop', methods=['POST'])
def stop_odia_listen():
    """Triggered when user taps the mic again to stop and process"""
    odia_mic_state["is_recording"] = False
    time.sleep(0.3) # Give the thread a split-second to flush the final audio chunks
    
    if not odia_mic_state["frames"]:
        return jsonify({"text": ""})
        
    # Stitch the chunks together
    audio_np = np.concatenate(odia_mic_state["frames"], axis=0)
    audio_np = np.squeeze(audio_np)
    
    # Normalize the waveform
    if np.max(np.abs(audio_np)) > 0:
        audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-9)
        
    # Run Inference
    inputs = stt_proc(audio_np, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = stt_model(inputs).logits
        
    pred_ids = torch.argmax(logits, dim=-1)
    text = stt_proc.batch_decode(pred_ids, skip_special_tokens=True)[0]
    
    return jsonify({"text": text})

@app.route('/api/tts/odia', methods=['POST'])
def tts_odia():
    """Takes Odia text and returns a playable WAV audio file."""
    text = request.json.get("text", "")
    
    inputs = tts_tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_audio = tts_model(**inputs).waveform[0].cpu().numpy()
        
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, rate=tts_model.config.sampling_rate, data=output_audio)
    wav_io.seek(0)
    
    return send_file(wav_io, mimetype="audio/wav")

# ==========================================
# 4. STANDARD ENDPOINTS
# ==========================================
@app.route('/')
def serve_frontend(): return send_file('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_assistant():
    data = request.json
    user_query = data.get('query', '').lower().strip()
    language = data.get('language', 'en')
    conn = get_db_connection()
    kb_items = conn.execute('SELECT * FROM knowledge_base WHERE language = ?', (language,)).fetchall()
    
    for item in kb_items:
        db_q = item['question'].lower()
        if db_q in user_query or user_query in db_q:
            conn.close()
            return jsonify({"answer": item['answer'], "source": "knowledge_base"})
            
    profile = conn.execute('SELECT * FROM business_profile LIMIT 1').fetchone()
    conn.close()
    biz_context = {"name": profile['name'] if profile else "Unknown", "type": profile['biz_type'] if profile else "Unknown"}
    ai_answer = generate_ai_fallback_response(user_query, biz_context)
    return jsonify({"answer": ai_answer, "source": "ai_fallback"})

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
    cursor.execute('INSERT INTO knowledge_base (question, answer, language) VALUES (?, ?, ?)', (data['question'], data['answer'], data['language']))
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
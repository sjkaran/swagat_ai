from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import os

app = Flask(__name__)
CORS(app)

# --- THE FIX: Force DB to save in the exact same folder as app.py ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "swagat_ai.db")
# --------------------------------------------------------------------

# ==========================================
# 1. DATABASE INITIALIZATION
# ==========================================
def init_db():
# ... rest of your code stays the same
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create Knowledge Base Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            language TEXT NOT NULL
        )
    ''')
    
    # Create Business Profile Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS business_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            biz_type TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

# Run this once when the app starts
if not os.path.exists(DB_NAME):
    init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Returns rows as dictionaries
    return conn

# ==========================================
# 2. AI FALLBACK MODULE (LEFT FOR YOU)
# ==========================================
def generate_ai_fallback_response(user_query, biz_context):
    """
    TODO: Integrate your Large Language Model (LLM) here.
    """
    # ... YOUR AI INTEGRATION GOES HERE ...
    
    return "I am currently unable to connect to my AI fallback brain. Please contact the staff directly."

# ==========================================
# 3. SERVE THE FRONTEND
# ==========================================
@app.route('/')
def serve_frontend():
    # This serves your index.html file at http://127.0.0.1:8000/
    return send_file('index.html')


# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.route('/api/ask', methods=['POST'])
def ask_assistant():
    data = request.json
    user_query = data.get('query', '').lower().strip()
    language = data.get('language', 'en')
    
    conn = get_db_connection()
    
    # 1. Check Knowledge Base
    kb_items = conn.execute(
        'SELECT * FROM knowledge_base WHERE language = ?', (language,)
    ).fetchall()
    
    for item in kb_items:
        db_question = item['question'].lower()
        if db_question in user_query or user_query in db_question:
            conn.close()
            return jsonify({"answer": item['answer'], "source": "knowledge_base"})
    
    # 2. If no match, gather context and trigger AI Fallback
    profile = conn.execute('SELECT * FROM business_profile LIMIT 1').fetchone()
    conn.close()
    
    biz_context = {
        "name": profile['name'] if profile else "Unknown",
        "type": profile['biz_type'] if profile else "Unknown"
    }
    
    ai_answer = generate_ai_fallback_response(data.get('query', ''), biz_context)
    
    return jsonify({"answer": ai_answer, "source": "ai_fallback"})

# --- Knowledge Base CRUD ---

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
    
    return jsonify({
        "id": new_id, 
        "question": data['question'], 
        "answer": data['answer'], 
        "language": data['language']
    })

@app.route('/api/kb/<int:item_id>', methods=['DELETE'])
def delete_kb_entry(item_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM knowledge_base WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Deleted successfully"})

# --- Business Profile CRUD ---

@app.route('/api/profile', methods=['GET'])
def get_profile():
    conn = get_db_connection()
    profile = conn.execute('SELECT * FROM business_profile LIMIT 1').fetchone()
    conn.close()
    
    if not profile:
        return jsonify({"name": "Odisha State Museum", "biz_type": "Heritage Site / Museum"})
    return jsonify(dict(profile))

@app.route('/api/profile', methods=['POST'])
def update_profile():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    
    profile = cursor.execute('SELECT id FROM business_profile LIMIT 1').fetchone()
    
    if not profile:
        cursor.execute(
            'INSERT INTO business_profile (name, biz_type) VALUES (?, ?)',
            (data['name'], data['biz_type'])
        )
    else:
        cursor.execute(
            'UPDATE business_profile SET name = ?, biz_type = ? WHERE id = ?',
            (data['name'], data['biz_type'], profile['id'])
        )
        
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Profile updated"})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
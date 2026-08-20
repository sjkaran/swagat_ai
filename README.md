<<<<<<< HEAD
# SwagatAI — स्वागत AI · ସ୍ୱାଗତ AI

**Self-trainable, voice-driven AI reception assistant for Odisha**

> Designed for hospitals, heritage sites, handicraft showrooms, gram panchayat offices, and any public-facing institution — with zero technical expertise required to operate.

---

## Quick Start

```bash
# 1. Clone or copy the project folder
cd swagatai

# 2. One-command install
bash install.sh

# 3. Activate environment and run
source venv/bin/activate
python run.py
```

Open your browser to `http://localhost:5000` — Swagata is live.

---

## Project Structure

```
swagatai/
├── run.py                  ← Launch server (python run.py)
├── install.sh              ← One-command setup script
├── frontend/
│   └── swagatai_frontend.html   ← UI (served by Flask)
├── backend/
│   ├── app.py              ← Flask application factory + all routes
│   ├── database.py         ← SQLAlchemy models
│   ├── kb_engine.py        ← Knowledge base search (fuzzy + n-gram)
│   ├── ai_fallback.py      ← Ollama LLM fallback + polite stubs
│   ├── config.py           ← Configuration (env-driven)
│   └── requirements.txt
├── data/
│   └── swagatai.db         ← SQLite database (auto-created)
└── logs/
    └── swagatai.log        ← Application log
```

---

## REST API Reference

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/api/health` | Server health + KB count |
| POST   | `/api/query`  | Submit visitor question → get answer |

### Knowledge Base

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/api/kb` | List all KB entries (filter: `?lang=or`) |
| POST   | `/api/kb` | Add a Q&A pair |
| PUT    | `/api/kb/<id>` | Update an entry |
| DELETE | `/api/kb/<id>` | Soft-delete an entry |
| POST   | `/api/kb/bulk` | Bulk import list of entries |

### Training & Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/api/suggestions`         | Unanswered queries pending training |
| POST   | `/api/suggestions/<id>/dismiss` | Mark suggestion as handled |
| GET    | `/api/analytics`           | Query counts, accuracy %, top questions |
| GET    | `/api/logs`                | Paginated query history |

### Business Profile

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/api/profile` | Get business profile |
| POST   | `/api/profile` | Save / update profile |

---

## Sample API Calls

```bash
# Ask a question (visitor)
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the timings?", "lang": "en"}'

# Add a training entry (owner)
curl -X POST http://localhost:5000/api/kb \
  -H "Content-Type: application/json" \
  -d '{"question": "Is wifi available?", "answer": "Yes, free wifi is available in the reception area.", "lang": "en"}'

# Get unanswered suggestions
curl http://localhost:5000/api/suggestions

# Analytics dashboard data
curl http://localhost:5000/api/analytics
```

---

## AI Fallback: Ollama (Offline LLM)

SwagatAI uses **Ollama** for local AI inference — no internet required at runtime.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the lightweight 3B model (~2 GB RAM, works on Raspberry Pi 4)
ollama pull llama3.2:3b

# For even lower RAM (Raspberry Pi 3 / 2 GB RAM):
ollama pull qwen2.5:1.5b
```

Set the model in environment:
```bash
OLLAMA_MODEL=qwen2.5:1.5b python run.py
```

**No Ollama?** The built-in polite stub automatically responds in Odia, Hindi, or English — visitors are never left with an error.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///data/swagatai.db` | Database connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model name |
| `SECRET_KEY` | (dev key) | Flask secret key — **change in production** |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `LOG_FILE` | `logs/swagatai.log` | Log file path |
| `DEBUG` | `false` | Enable Flask debug mode |

---

## Hardware Requirements

| Hardware | Works? | Notes |
|----------|--------|-------|
| Raspberry Pi 4 (4 GB) | ✅ | Ollama with 3B model |
| Raspberry Pi 4 (2 GB) | ✅ | Use 1.5B model |
| Raspberry Pi 3 | ✅ | Stub mode only (no Ollama) |
| Old laptop / PC | ✅ | Full Ollama support |
| Android tablet + Termux | ✅ | Stub + KB only |

---

## Languages Supported

| Language | Code | Script |
|----------|------|--------|
| Odia | `or` | ଓଡ଼ିଆ |
| Hindi | `hi` | देवनागरी |
| English | `en` | Latin |

The KB engine uses **Unicode NFKC normalisation + character n-grams** for script-aware matching, so Odia and Hindi queries match correctly without any romanisation.

---

## Deployment as System Service (Linux)

```bash
# Create systemd service
sudo nano /etc/systemd/system/swagatai.service
```

```ini
[Unit]
Description=SwagatAI Reception Assistant
After=network.target

[Service]
WorkingDirectory=/home/pi/swagatai
ExecStart=/home/pi/swagatai/venv/bin/python run.py --port 80
Restart=always
User=pi
Environment=SECRET_KEY=your-production-secret

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable swagatai
sudo systemctl start swagatai
```

---

## Training the Assistant (No Tech Skills Needed)

1. Open `http://localhost:5000` in any browser
2. Switch to **Owner Mode** (top-right toggle)
3. Go to the **Train** tab on the right panel
4. Type (or speak) a question a visitor might ask
5. Type the correct answer
6. Click **Add to Knowledge Base**
7. The assistant learns instantly — no restart needed

The **Suggestions** tab automatically shows questions visitors asked that Swagata couldn't answer — one click to fill in the training answer.

---

## Built With

- **Python 3.11+** · Flask · SQLAlchemy · SQLite
- **Frontend**: Vanilla HTML/CSS/JS · Web Speech API · SVG animation
- **AI**: Ollama (local LLM) · Built-in multilingual stubs
- **Search**: Custom fuzzy KB engine (token Jaccard + character n-grams)

---

*SwagatAI — Made for Odisha, built for every business.*  
*ଓଡ଼ିଶା ପାଇଁ ତିଆରି · ہر کاروبار کے لیے*
=======
# swagat_ai
"SwagatAI" (ସ୍ୱାଗତ AI) Odisha's First Trainable Vernacular Voice Reception Assistant  A trainable AI receptionist that any Odisha business — hospital, government office, tourist site, handicraft store — can teach in Odia/Hindi/English to handle customer queries verbally, without hiring extra staff.



## Tech stack

*Voice Engine*
> STT - Whisper, TTS - win32com & pythoncom, 
> recording - Edge-web-recorder (depends on internet connection)

*Backend*
> Python, flask, 
> AI - Ollama

*DB*
> Sqlite, 
> Json

*Front End:*
> Html, 
> CSS, 
> Java Script



### Vision:

*"To make this model efficient enough to be in the use of common public of Odisha and India."*
>>>>>>> 7fb4ba510767b55c2b2e54b33566cc8c073b54e2

"""
config.py — Configuration for SwagatAI backend
"""

import os


class Config:
    # ── Flask ─────────────────────────────────────────────────────────────
    SECRET_KEY   = os.environ.get('SECRET_KEY', 'swagatai-dev-secret-change-in-prod')
    DEBUG        = os.environ.get('DEBUG', 'false').lower() == 'true'

    # ── Database ──────────────────────────────────────────────────────────
    _base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(_base_dir, "data", "swagatai.db")}'
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')

    # ── AI Fallback (Ollama) ──────────────────────────────────────────────
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL    = os.environ.get('OLLAMA_MODEL',    'llama3.2:3b')

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_FILE = os.environ.get(
        'LOG_FILE',
        os.path.join(_base_dir, 'logs', 'swagatai.log')
    )


class ProductionConfig(Config):
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True

"""
database.py — SQLAlchemy models for SwagatAI
"""

import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class KBEntry(db.Model):
    """A single Q&A pair in the knowledge base."""
    __tablename__ = 'kb_entries'

    id          = db.Column(db.Integer, primary_key=True)
    question    = db.Column(db.Text, nullable=False)
    answer      = db.Column(db.Text, nullable=False)
    lang        = db.Column(db.String(8), nullable=False, default='en')  # 'en'|'hi'|'or'
    tags        = db.Column(db.Text, default='[]')                       # JSON list
    hit_count   = db.Column(db.Integer, default=0)
    active      = db.Column(db.Boolean, default=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id':         self.id,
            'question':   self.question,
            'answer':     self.answer,
            'lang':       self.lang,
            'tags':       json.loads(self.tags or '[]'),
            'hit_count':  self.hit_count,
            'active':     self.active,
            'created_at': self.created_at.isoformat(),
        }

    def __repr__(self):
        return f'<KBEntry {self.id} [{self.lang}] {self.question[:40]!r}>'


class QueryLog(db.Model):
    """Every visitor query, its answer, and its source."""
    __tablename__ = 'query_logs'

    id          = db.Column(db.Integer, primary_key=True)
    question    = db.Column(db.Text, nullable=False)
    answer      = db.Column(db.Text, nullable=False)
    lang        = db.Column(db.String(8), default='en')
    source      = db.Column(db.String(20), default='kb')  # 'kb' | 'ai_fallback'
    confidence  = db.Column(db.Float, default=1.0)
    session_id  = db.Column(db.String(64), default='anonymous')
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':         self.id,
            'question':   self.question,
            'answer':     self.answer,
            'lang':       self.lang,
            'source':     self.source,
            'confidence': self.confidence,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
        }


class UnansweredLog(db.Model):
    """Queries not found in KB — surfaced as training suggestions."""
    __tablename__ = 'unanswered_logs'

    id          = db.Column(db.Integer, primary_key=True)
    question    = db.Column(db.Text, nullable=False)
    lang        = db.Column(db.String(8), default='en')
    session_id  = db.Column(db.String(64), default='anonymous')
    count       = db.Column(db.Integer, default=1)
    trained     = db.Column(db.Boolean, default=False)   # True = owner acted on it
    last_seen   = db.Column(db.DateTime, default=datetime.utcnow)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':        self.id,
            'question':  self.question,
            'lang':      self.lang,
            'count':     self.count,
            'trained':   self.trained,
            'last_seen': self.last_seen.isoformat(),
        }


class BusinessProfile(db.Model):
    """Business / institution details (single record per deployment)."""
    __tablename__ = 'business_profile'

    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(200), default='My Business')
    type        = db.Column(db.String(100), default='General')
    address     = db.Column(db.Text, default='')
    phone       = db.Column(db.String(20), default='')
    languages   = db.Column(db.Text, default='["en"]')  # JSON list
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id':        self.id,
            'name':      self.name,
            'type':      self.type,
            'address':   self.address,
            'phone':     self.phone,
            'languages': json.loads(self.languages or '["en"]'),
        }

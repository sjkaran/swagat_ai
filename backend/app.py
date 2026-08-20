"""
SwagatAI Backend — app.py
Flask REST API for the SwagatAI reception assistant.
Handles KB management, query resolution, AI fallback, logging, and voice proxy.
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from database import db, KBEntry, QueryLog, UnansweredLog, BusinessProfile
from kb_engine import KBEngine
from ai_fallback import AIFallback
from config import Config

# ─── APP SETUP ───────────────────────────────────────────────────────────────

def create_app(config: Config = None) -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'),
        static_url_path='',
    )

    cfg = config or Config()
    app.config.from_object(cfg)

    # SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = cfg.DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Extensions
    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": cfg.ALLOWED_ORIGINS}})

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(cfg.LOG_FILE),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger('swagatai')

    # Services
    kb_engine = KBEngine()
    ai_fallback = AIFallback(model=cfg.OLLAMA_MODEL, base_url=cfg.OLLAMA_BASE_URL)

    # Create tables
    with app.app_context():
        db.create_all()
        _seed_demo_data()

    # ─── ROUTES ──────────────────────────────────────────────────────────────

    # Serve frontend
    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'swagatai_frontend.html')

    # ── Health ──────────────────────────────────────────────────────────────
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'ok',
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat(),
            'kb_count': KBEntry.query.filter_by(active=True).count(),
            'ai_fallback_available': ai_fallback.is_available(),
        })

    # ── Query (core visitor endpoint) ────────────────────────────────────────
    @app.route('/api/query', methods=['POST'])
    def handle_query():
        """
        Main endpoint: visitor asks a question.
        1. Search knowledge base (fuzzy + semantic).
        2. If found → return trained answer.
        3. If not found → try AI fallback (Ollama / stub).
        4. Log everything for training suggestions.
        """
        data = request.get_json(silent=True) or {}
        question  = (data.get('question') or '').strip()
        lang      = data.get('lang', 'en')
        session_id = data.get('session_id', 'anonymous')

        if not question:
            return jsonify({'error': 'question is required'}), 400

        logger.info(f"[QUERY] lang={lang} session={session_id} q={question!r}")

        # 1. KB lookup
        kb_match = kb_engine.search(question, lang=lang)

        if kb_match:
            answer     = kb_match['answer']
            source     = 'kb'
            confidence = kb_match['score']
            used_ai    = False

            # Increment hit count
            entry = KBEntry.query.get(kb_match['id'])
            if entry:
                entry.hit_count += 1
                db.session.commit()
        else:
            # 2. AI fallback
            ai_result  = ai_fallback.answer(question, lang=lang)
            answer     = ai_result['answer']
            source     = 'ai_fallback'
            confidence = ai_result.get('confidence', 0.5)
            used_ai    = True

            # Log as unanswered for training suggestion
            _log_unanswered(question, lang, session_id)

        # 3. Persist query log
        log = QueryLog(
            question=question,
            answer=answer,
            lang=lang,
            source=source,
            confidence=confidence,
            session_id=session_id,
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            'answer':     answer,
            'source':     source,
            'confidence': round(confidence, 3),
            'used_ai':    used_ai,
            'log_id':     log.id,
        })

    # ── Knowledge Base CRUD ─────────────────────────────────────────────────
    @app.route('/api/kb', methods=['GET'])
    def list_kb():
        lang_filter = request.args.get('lang')
        q = KBEntry.query.filter_by(active=True)
        if lang_filter:
            q = q.filter_by(lang=lang_filter)
        entries = q.order_by(KBEntry.created_at.desc()).all()
        return jsonify({'entries': [e.to_dict() for e in entries], 'count': len(entries)})

    @app.route('/api/kb', methods=['POST'])
    def add_kb():
        data = request.get_json(silent=True) or {}
        question = (data.get('question') or '').strip()
        answer   = (data.get('answer')   or '').strip()
        lang     = data.get('lang', 'en')
        tags     = data.get('tags', [])

        if not question or not answer:
            return jsonify({'error': 'question and answer are required'}), 400

        entry = KBEntry(
            question=question,
            answer=answer,
            lang=lang,
            tags=json.dumps(tags),
        )
        db.session.add(entry)
        db.session.commit()
        logger.info(f"[KB ADD] id={entry.id} lang={lang} q={question!r}")

        # Rebuild engine index
        kb_engine.rebuild(KBEntry.query.filter_by(active=True).all())

        return jsonify({'entry': entry.to_dict(), 'message': 'Added successfully'}), 201

    @app.route('/api/kb/<int:entry_id>', methods=['PUT'])
    def update_kb(entry_id):
        entry = KBEntry.query.get_or_404(entry_id)
        data  = request.get_json(silent=True) or {}
        if 'question' in data: entry.question = data['question'].strip()
        if 'answer'   in data: entry.answer   = data['answer'].strip()
        if 'lang'     in data: entry.lang     = data['lang']
        if 'tags'     in data: entry.tags     = json.dumps(data['tags'])
        entry.updated_at = datetime.utcnow()
        db.session.commit()
        kb_engine.rebuild(KBEntry.query.filter_by(active=True).all())
        return jsonify({'entry': entry.to_dict()})

    @app.route('/api/kb/<int:entry_id>', methods=['DELETE'])
    def delete_kb(entry_id):
        entry = KBEntry.query.get_or_404(entry_id)
        entry.active = False           # soft-delete
        db.session.commit()
        kb_engine.rebuild(KBEntry.query.filter_by(active=True).all())
        return jsonify({'message': 'Deleted', 'id': entry_id})

    @app.route('/api/kb/bulk', methods=['POST'])
    def bulk_import():
        """Import multiple Q&A pairs at once (CSV / JSON batch)."""
        data  = request.get_json(silent=True) or {}
        items = data.get('entries', [])
        added = 0
        for item in items:
            q = (item.get('question') or '').strip()
            a = (item.get('answer')   or '').strip()
            l = item.get('lang', 'en')
            if q and a:
                db.session.add(KBEntry(question=q, answer=a, lang=l))
                added += 1
        db.session.commit()
        kb_engine.rebuild(KBEntry.query.filter_by(active=True).all())
        return jsonify({'added': added}), 201

    # ── Business Profile ────────────────────────────────────────────────────
    @app.route('/api/profile', methods=['GET'])
    def get_profile():
        profile = BusinessProfile.query.first()
        if not profile:
            return jsonify({'profile': {}})
        return jsonify({'profile': profile.to_dict()})

    @app.route('/api/profile', methods=['POST', 'PUT'])
    def save_profile():
        data = request.get_json(silent=True) or {}
        profile = BusinessProfile.query.first()
        if not profile:
            profile = BusinessProfile()
            db.session.add(profile)
        profile.name        = data.get('name', profile.name or 'My Business')
        profile.type        = data.get('type', profile.type or 'General')
        profile.address     = data.get('address', profile.address or '')
        profile.phone       = data.get('phone', profile.phone or '')
        profile.languages   = json.dumps(data.get('languages', ['or', 'hi', 'en']))
        profile.updated_at  = datetime.utcnow()
        db.session.commit()
        return jsonify({'profile': profile.to_dict()})

    # ── Unanswered / Suggestions ────────────────────────────────────────────
    @app.route('/api/suggestions', methods=['GET'])
    def get_suggestions():
        items = UnansweredLog.query.filter_by(trained=False)\
                    .order_by(UnansweredLog.count.desc())\
                    .limit(50).all()
        return jsonify({'suggestions': [s.to_dict() for s in items]})

    @app.route('/api/suggestions/<int:sid>/dismiss', methods=['POST'])
    def dismiss_suggestion(sid):
        s = UnansweredLog.query.get_or_404(sid)
        s.trained = True
        db.session.commit()
        return jsonify({'message': 'Dismissed'})

    # ── Analytics ───────────────────────────────────────────────────────────
    @app.route('/api/analytics', methods=['GET'])
    def analytics():
        total     = QueryLog.query.count()
        kb_hits   = QueryLog.query.filter_by(source='kb').count()
        ai_hits   = QueryLog.query.filter_by(source='ai_fallback').count()
        accuracy  = round(kb_hits / total * 100, 1) if total else 0

        # Top 5 questions
        from sqlalchemy import func
        top_q = db.session.query(
            QueryLog.question,
            func.count(QueryLog.question).label('cnt')
        ).group_by(QueryLog.question)\
         .order_by(func.count(QueryLog.question).desc())\
         .limit(5).all()

        return jsonify({
            'total_queries': total,
            'kb_answered':   kb_hits,
            'ai_answered':   ai_hits,
            'accuracy_pct':  accuracy,
            'top_questions': [{'question': r[0], 'count': r[1]} for r in top_q],
            'kb_size':       KBEntry.query.filter_by(active=True).count(),
        })

    # ── Logs ────────────────────────────────────────────────────────────────
    @app.route('/api/logs', methods=['GET'])
    def get_logs():
        page     = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        lang_f   = request.args.get('lang')
        q        = QueryLog.query
        if lang_f:
            q = q.filter_by(lang=lang_f)
        pag = q.order_by(QueryLog.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
        return jsonify({
            'logs':    [l.to_dict() for l in pag.items],
            'total':   pag.total,
            'pages':   pag.pages,
            'page':    page,
        })

    # ── Error handlers ──────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Server error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

    return app


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _seed_demo_data():
    """Seed the DB with demo KB entries if empty."""
    if KBEntry.query.count() > 0:
        return
    demo = [
        ("What are the opening hours?",
         "We are open Tuesday to Sunday, 10 AM to 5 PM. Closed on Mondays and public holidays.",
         "en", ["timing", "hours"]),
        ("What is the ticket price?",
         "Tickets: Indian adults ₹20, children (below 15) free, foreign nationals ₹200.",
         "en", ["ticket", "price"]),
        ("ଟିକେଟ୍ ଦର କେତେ?",
         "ଭାରତୀୟ ବୟସ୍କ: ₹୨୦, ୧୫ ବର୍ଷ ତଳ ଶିଶୁ: ମାଗଣା, ବିଦେଶୀ: ₹୨୦୦।",
         "or", ["ticet", "dara"]),
        ("ସମୟ ସୂଚୀ?",
         "ମଙ୍ଗଳବାର ରୁ ରବିବାର ସକାଳ ୧୦ ଟାରୁ ବିକାଳ ୫ ଟା ପର୍ଯ୍ୟନ୍ତ। ସୋମବାର ବନ୍ଦ।",
         "or", ["samay"]),
        ("टिकट कितने का है?",
         "भारतीय वयस्क: ₹20, 15 वर्ष से कम बच्चे: निःशुल्क, विदेशी: ₹200।",
         "hi", ["ticket", "price"]),
        ("खुलने का समय क्या है?",
         "मंगलवार से रविवार, सुबह 10 बजे से शाम 5 बजे तक। सोमवार को बंद।",
         "hi", ["timing"]),
        ("Is photography allowed?",
         "Yes, non-flash photography is allowed in most galleries. Flash photography and tripods are prohibited inside.",
         "en", ["photography", "camera"]),
        ("Where is the parking?",
         "Free parking is available adjacent to Gate No. 1. Capacity: 50 cars and 100 two-wheelers.",
         "en", ["parking"]),
        ("Is there a cafeteria or restaurant?",
         "Yes, a canteen is available near Gate 2, open 10 AM to 4 PM serving snacks and meals.",
         "en", ["food", "cafeteria", "canteen"]),
        ("How do I reach the museum?",
         "We are located on Lewis Road, Bhubaneswar. Nearest bus stop: Museum Square (routes 7, 12, 25). Auto-rickshaws available from Bhubaneswar Railway Station (4 km).",
         "en", ["location", "direction", "reach"]),
    ]
    for q, a, lang, tags in demo:
        db.session.add(KBEntry(question=q, answer=a, lang=lang, tags=json.dumps(tags)))
    db.session.add(BusinessProfile(
        name="Odisha State Museum",
        type="Heritage Site / Museum",
        address="Lewis Road, Bhubaneswar, Odisha 751014",
        phone="+91-674-2432397",
        languages=json.dumps(["or", "hi", "en"]),
    ))
    db.session.commit()


def _log_unanswered(question: str, lang: str, session_id: str):
    """Deduplicate and track unanswered queries for training suggestions."""
    existing = UnansweredLog.query.filter_by(question=question, trained=False).first()
    if existing:
        existing.count += 1
        existing.last_seen = datetime.utcnow()
    else:
        db.session.add(UnansweredLog(question=question, lang=lang, session_id=session_id))
    db.session.commit()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)

#!/usr/bin/env python3
"""
run.py — Launch SwagatAI backend server
Usage:
    python run.py              # default: http://0.0.0.0:5000
    python run.py --port 8080
    python run.py --debug
"""

import argparse
import os
import sys

# Ensure backend/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import create_app
from backend.config import DevelopmentConfig, ProductionConfig


def main():
    parser = argparse.ArgumentParser(description='SwagatAI Reception Assistant Server')
    parser.add_argument('--host',  default='0.0.0.0',  help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port',  default=5000, type=int, help='Port (default: 5000)')
    parser.add_argument('--debug', action='store_true',   help='Enable debug mode')
    args = parser.parse_args()

    cfg = DevelopmentConfig() if args.debug else ProductionConfig()
    app = create_app(cfg)

    # Ensure data/logs directories exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)

    print(f"""
╔══════════════════════════════════════════╗
║         SwagatAI Backend v1.0            ║
║   ସ୍ୱାଗତ  •  स्वागत  •  Welcome          ║
╠══════════════════════════════════════════╣
║  http://{args.host}:{args.port:<4}                      ║
║  Debug : {str(args.debug):<33}║
╚══════════════════════════════════════════╝
""")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

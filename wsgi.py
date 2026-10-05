"""Production WSGI entry point for Fake News Detection App.

Supports:
  - Linux/Container: gunicorn -c gunicorn_config.py wsgi:app
  - Windows: python wsgi.py  (runs with Waitress production WSGI server)
"""
import sys
from app import app
from config.settings import Config

if __name__ == "__main__":
    if sys.platform == "win32":
        # Windows does not support gunicorn natively due to missing fcntl/posix calls.
        # Waitress is the recommended production multi-threaded WSGI server on Windows.
        from waitress import serve
        print(f"[*] Starting production Waitress WSGI server on http://{Config.HOST}:{Config.PORT}...")
        serve(app, host=Config.HOST, port=Config.PORT, threads=6)
    else:
        # On Linux/macOS, direct execution or gunicorn
        print(f"[*] Starting server on http://{Config.HOST}:{Config.PORT}...")
        app.run(host=Config.HOST, port=Config.PORT)

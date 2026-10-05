"""Gunicorn production configuration file."""
import multiprocessing, os

bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '5000')}"
workers = int(os.getenv("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
timeout = int(os.getenv("TIMEOUT", "30"))
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")

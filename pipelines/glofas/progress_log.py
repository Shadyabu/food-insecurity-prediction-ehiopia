"""
progress_log.py

Shared progress-logging helper for the GloFAS backfill run. Appends
timestamped entries to BACKFILL_PROGRESS.md so the run's status is
checkable anytime by just opening the file -- a single fetch script
invocation covers many sequential requests and won't return control until
the whole run finishes, so this is the way to see live progress without
waiting on a background-task completion notification.
"""

import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, "BACKFILL_PROGRESS.md")


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"- `{timestamp}` {message}\n")
    print(message)


def log_header(title):
    with open(LOG_PATH, "a") as f:
        f.write(f"\n## {title}\n\n")

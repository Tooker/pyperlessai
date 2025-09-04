#!/usr/bin/env python3
"""
Scheduler for the PoC using APScheduler.

Reads:
 - CRON_SCHEDULE (crontab string, default "*/15 * * * *")
 - POC_LIMIT (int, default 1)
 - RUN_ON_START (bool, default True)

Executes the PoC as a subprocess to avoid importing src/poc.py directly
(which performs env checks at import-time).
Logs to stdout so container logs capture job output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import signal
from typing import Any
from loguru import logger

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception as e:
    logger.exception(f"Failed to import APScheduler: {e}")
    raise

# Environment / defaults
CRON_SCHEDULE = os.getenv("CRON_SCHEDULE", "*/15 * * * *")
POC_LIMIT = int(os.getenv("POC_LIMIT", "1"))
RUN_ON_START = os.getenv("RUN_ON_START", "true").lower() in ("1", "true", "yes")

# Ensure unbuffered output when subprocess runs python -u; still set PYTHONUNBUFFERED here for direct runs.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"), format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}")

def run_poc(limit: int = POC_LIMIT) -> None:
    """
    Run the PoC as a subprocess and stream its output to this process' stdout/stderr.
    """
    cmd = ["python", "-u", "src/poc.py", "--limit", str(limit)]
    logger.info(f"Starting PoC subprocess: {cmd}")
    try:
        result = subprocess.run(cmd)
        logger.info(f"PoC subprocess exited with returncode={result.returncode}")
    except Exception as e:
        logger.exception(f"Error while running PoC subprocess: {e}")

def create_scheduler(cron_expr: str) -> BlockingScheduler:
    """
    Create and return a BlockingScheduler configured with the supplied crontab string.
    """
    sched = BlockingScheduler()
    trigger = CronTrigger.from_crontab(cron_expr)
    sched.add_job(run_poc, trigger=trigger, id="poc_job", replace_existing=True)
    logger.info(f"Scheduled PoC with cron expression: '{cron_expr}' (job id: poc_job)")
    return sched

def _handle_shutdown(signum: int, frame: Any) -> None:
    logger.info(f"Received signal {signum}; shutting down.")
    # APScheduler will be stopped in the main try/except finally by calling shutdown()
    # We just exit; main will catch KeyboardInterrupt/SystemExit.
    raise KeyboardInterrupt()

def main() -> None:
    # Wire up signals for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    logger.info(f"Scheduler starting (RUN_ON_START={RUN_ON_START}, POC_LIMIT={POC_LIMIT})")

    if RUN_ON_START:
        logger.info("RUN_ON_START is true — running PoC once at startup.")
        run_poc(POC_LIMIT)

    sched = create_scheduler(CRON_SCHEDULE)

    try:
        logger.info("Starting APScheduler (blocking).")
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopping due to interrupt.")
    finally:
        try:
            sched.shutdown(wait=False)
            logger.info("Scheduler shut down.")
        except Exception:
            pass

if __name__ == "__main__":
    main()

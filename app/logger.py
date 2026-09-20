import logging
import os
from logging.handlers import RotatingFileHandler

LOG_PATH = os.path.join(os.getcwd(), "logs", "app.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5_000_000,
        backupCount=3
    )

    handler.setFormatter(logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","message":%(message)s}'
    ))

    logger.addHandler(handler)

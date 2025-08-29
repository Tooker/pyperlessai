#!/usr/bin/env python3
"""
PoC: Fetch thumbnails from paperless-ngx, normalize and send to OpenAI (vision) for analysis.

This module now contains only orchestration code. Heavy lifting is delegated to:
 - src/paperless.py (PaperlessClient)
 - src/ai_client.py   (AIClient)
"""

import os
import sys
import argparse
import asyncio
import io
from loguru import logger
from dotenv import load_dotenv
import yaml
import json
from typing import Optional, Dict, Any

# Load environment variables from a local .env file (if present)
load_dotenv()

# Load settings.yaml from project root (optional). If not found, fall back to defaults.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.yaml")
try:
    with open(SETTINGS_FILE, "r", encoding="utf-8") as sf:
        _settings = yaml.safe_load(sf) or {}
except FileNotFoundError:
    _settings = {}
# Extract prompt (multi-line string) or None
SETTINGS_PROMPT = _settings.get("prompt")

# Configuration via ENV (kept same names for backward compatibility)
PAPERLESS_BASE_URL = os.getenv("PAPERLESS_BASE_URL")
PAPERLESS_API_TOKEN = os.getenv("PAPERLESS_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGESize".upper(), os.getenv("MAX_IMAGE_SIZE", "1024")))  # px
CONCURRENT_WORKERS = int(os.getenv("CONCURRENT_WORKERS", "4"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Use loguru for structured logging
logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
)

if not PAPERLESS_BASE_URL:
    logger.error("Please set PAPERLESS_BASE_URL environment variable.")
    sys.exit(1)
if not PAPERLESS_API_TOKEN:
    logger.error("Please set PAPERLESS_API_TOKEN environment variable.")
    sys.exit(1)
if not OPENAI_API_KEY:
    logger.error("Please set OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Import refactored clients
from ai_client import AIClient
from classifier import PaperlessOpenAIClassifier


async def process_document(
    semaphore: asyncio.Semaphore,
    ai: AIClient,
    classifier: PaperlessOpenAIClassifier,
    doc: Dict[str, Any],
    max_image_size: int,
    out_dir: str = "poc_output",
):
    async with semaphore:
        doc_id = doc.get("id") or doc.get("pk") or doc.get("document_id")
        title = doc.get("title") or doc.get("name") or "<untitled>"
        logger.info(f"Processing doc id={doc_id} title={title}")
        if not doc_id:
            logger.warning(f"No id found on document: {doc}")
            return

        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"Analyzing Paperless document id={doc_id} with classifier")
        try:
            # Classifier now handles fetching the document bytes via the provided PaperlessClient.
            result = await classifier.analyze_paperless_document(doc_id, extra_instructions=None)

            # Convert result to JSON for logging/saving in a robust way that handles SDK objects or dicts.
            raw_json = ""
            try:
                if hasattr(result, "model_dump_json"):
                    raw_json = result.model_dump_json(indent=2)
                else:
                    raw_json = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception:
                try:
                    raw_json = json.dumps({"raw": str(result)}, ensure_ascii=False, indent=2)
                except Exception:
                    raw_json = str(result)

            json_out = os.path.join(out_dir, f"{doc_id}.openai.json")
            json_parsed_out = os.path.join(out_dir, f"{doc_id}.openai_parsed.json")

            with open(json_out, "w", encoding="utf-8") as jf:
                jf.write(raw_json)

            # Attempt to save parsed output if available on the SDK object or dict
            try:
                if hasattr(result, "output_parsed"):
                    parsed_json = result.output_parsed.model_dump_json(indent=2)
                    with open(json_parsed_out, "w", encoding="utf-8") as parsed:
                        parsed.write(parsed_json)
                elif isinstance(result, dict) and "output_parsed" in result:
                    with open(json_parsed_out, "w", encoding="utf-8") as parsed:
                        parsed.write(json.dumps(result["output_parsed"], ensure_ascii=False, indent=2))
            except Exception:
                # Ignore parsing save errors; main result is already saved.
                pass

            logger.info(f"Saved OpenAI JSON result to {json_out}")
        except Exception as e:
            logger.exception("Failed to call classifier for doc %s: %s", doc_id, e)
        return


async def main(limit: int = 10):
    ai = AIClient(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, request_timeout=120)

    try:
        # Create classifier which owns its PaperlessClient internally and run the end-to-end flow.
        classifier = PaperlessOpenAIClassifier(ai=ai, base_url=PAPERLESS_BASE_URL, api_token=PAPERLESS_API_TOKEN, base_prompt=SETTINGS_PROMPT)
        await classifier.run(
            limit=limit,
            concurrent_workers=CONCURRENT_WORKERS,
            max_image_size=MAX_IMAGE_SIZE,
            out_dir="poc_output",
        )
    except Exception as e:
        logger.exception("Unexpected error in main: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoC: Fetch paperless-ngx thumbnails -> OpenAI (refactored)")
    parser.add_argument(
        "--limit", type=int, default=1, help="Maximum number of documents to process"
    )
    
    args = parser.parse_args()
    try:
        asyncio.run(main(limit=args.limit))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

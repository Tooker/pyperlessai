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
from paperless import PaperlessClient
from ai_client import AIClient


async def process_document(
    semaphore: asyncio.Semaphore,
    paperless: PaperlessClient,
    ai: AIClient,
    http_client,
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

        content = await paperless.fetch_document_bytes(http_client, doc_id)
        if not content:
            logger.warning(f"No thumbnail/download available for doc {doc_id}")
            return

        os.makedirs(out_dir, exist_ok=True)

        # If PDF, upload PDF to OpenAI
        if content.startswith(b"%PDF"):
            logger.info(f"Downloaded content is PDF for doc {doc_id} — uploading to OpenAI")
            prompt_text = SETTINGS_PROMPT or "Please extract any text (OCR) and list relevant labels and a short caption describing the document."
            try:
                result = await ai.send_pdf_bytes(content, prompt_text)
                # Convert SDK object to serializable dict if possible
                try:
                    if hasattr(result, "to_dict"):
                        result_obj = result.to_dict()
                    elif isinstance(result, dict):
                        result_obj = result
                    else:
                        result_obj = {"raw": str(result)}
                except Exception:
                    result_obj = {"raw": str(result)}

                json_out = os.path.join(out_dir, f"{doc_id}.openai.json")
                import json

                with open(json_out, "w", encoding="utf-8") as jf:
                    json.dump(result_obj, jf, ensure_ascii=False, indent=2)
                logger.info(f"Saved OpenAI JSON result to {json_out}")
            except Exception as e:
                logger.exception("Failed to call OpenAI for PDF doc %s: %s", doc_id, e)
            return

        # Otherwise normalize image and send to OpenAI
        norm = AIClient.normalize_image_bytes(content, max_size=max_image_size)
        out_path = os.path.join(out_dir, f"{doc_id}.jpg")
        with open(out_path, "wb") as f:
            f.write(norm)
        logger.info(f"Saved normalized image to {out_path}")

        prompt_text = SETTINGS_PROMPT or "Please extract any text (OCR) and list relevant labels and a short caption describing the image."
        try:
            result = await ai.send_image_bytes(http_client, norm, prompt_text)
            # Save full JSON for inspection
            json_out = os.path.join(out_dir, f"{doc_id}.openai.json")
            import json

            with open(json_out, "w", encoding="utf-8") as jf:
                json.dump(result, jf, ensure_ascii=False, indent=2)
            logger.info(f"Saved OpenAI JSON result to {json_out}")
            logger.info(f"OpenAI result for doc {doc_id}: {str(result)[:1000]}")
        except Exception as e:
            logger.exception("Failed to call OpenAI for doc %s: %s", doc_id, e)


async def main(limit: int = 10, page_size: int = 500):
    paperless = PaperlessClient(base_url=PAPERLESS_BASE_URL, api_token=PAPERLESS_API_TOKEN, request_timeout=REQUEST_TIMEOUT)
    ai = AIClient(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, request_timeout=120)

    # Single shared httpx client for paperless fetching and image-based OpenAI requests
    headers = {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}
    timeout = httpx_timeout = None
    try:
        import httpx
        timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as http_client:
            docs = await paperless.list_documents(http_client, page_size=page_size, limit=limit)
            logger.info(f"Found {len(docs)} documents (requested limit={limit})")

            # Also fetch auxiliary interfaces: tags, document types, correspondents
            out_dir = "poc_output"
            os.makedirs(out_dir, exist_ok=True)

            try:
                tags = await paperless.list_tags(http_client)
                with open(os.path.join(out_dir, "tags.json"), "w", encoding="utf-8") as jf:
                    json.dump(tags, jf, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(tags)} tags to {os.path.join(out_dir, 'tags.json')}")
            except Exception as e:
                logger.exception("Failed to fetch/save tags: %s", e)

            try:
                document_types = await paperless.list_document_types(http_client)
                with open(os.path.join(out_dir, "document_types.json"), "w", encoding="utf-8") as jf:
                    json.dump(document_types, jf, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(document_types)} document types to {os.path.join(out_dir, 'document_types.json')}")
            except Exception as e:
                logger.exception("Failed to fetch/save document types: %s", e)

            try:
                correspondents = await paperless.list_correspondents(http_client)
                with open(os.path.join(out_dir, "correspondents.json"), "w", encoding="utf-8") as jf:
                    json.dump(correspondents, jf, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(correspondents)} correspondents to {os.path.join(out_dir, 'correspondents.json')}")
            except Exception as e:
                logger.exception("Failed to fetch/save correspondents: %s", e)

            semaphore = asyncio.Semaphore(CONCURRENT_WORKERS)
            tasks = []
            for doc in docs:
                tasks.append(process_document(semaphore, paperless, ai, http_client, doc, MAX_IMAGE_SIZE))
            await asyncio.gather(*tasks)
    except Exception as e:
        logger.exception("Unexpected error in main: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoC: Fetch paperless-ngx thumbnails -> OpenAI (refactored)")
    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of documents to process"
    )
    parser.add_argument(
        "--page_size", type=int, default=500, help="Page size when listing documents"
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(limit=args.limit, page_size=args.page_size))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

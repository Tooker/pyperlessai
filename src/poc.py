#!/usr/bin/env python3
"""
PoC: Fetch thumbnails from paperless-ngx, normalize and send to OpenAI (vision) for analysis.

Usage:
  - Set environment variables:
      PAPERLESS_BASE_URL (e.g. https://paperless.example)
      PAPERLESS_API_TOKEN (API token for paperless-ngx)
      OPENAI_API_KEY
      OPENAI_MODEL (optional, default: gpt-4o-mini-vision)
  - Run:
      python src/poc.py --limit 10

Notes / Warnings:
  - paperless-ngx API paths vary by version. This PoC tries common endpoints:
      GET {base}/api/documents/        -> list (expects 'results' / 'next')
      GET {base}/api/documents/{id}/thumbnail/  -> thumbnail (preferred)
      GET {base}/api/documents/{id}/download/   -> full document (PDF) fallback
    Adjust endpoints if your instance differs.
  - This example calls OpenAI Responses endpoint with an inline base64 image. Adapt if your OpenAI client or endpoint differs.
  - Do NOT hardcode secrets. Use environment variables or a secret manager.
"""

import os
import sys
import argparse
import asyncio
import base64
import io
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from a local .env file (if present)
load_dotenv()
from typing import Optional, List, Dict

import httpx
from PIL import Image

# Configuration via ENV
PAPERLESS_BASE_URL = os.getenv("PAPERLESS_BASE_URL")
PAPERLESS_API_TOKEN = os.getenv("PAPERLESS_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))  # px
CONCURRENT_WORKERS = int(os.getenv("CONCURRENT_WORKERS", "4"))
REQUEST_TIMEOUT = 30

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

import openai

# Create a single global OpenAI client instance for the whole module.
# Prefer the class-based client if available, otherwise fall back to the module-level API object.
try:
    from openai import OpenAI as _OpenAI

    OPENAI_CLIENT = _OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    OPENAI_CLIENT = openai
# Also set module-level api_key for compatibility with older callsites
try:
    openai.api_key = OPENAI_API_KEY
except Exception:
    pass


def u(img_bytes: bytes, max_size: int = MAX_IMAGE_SIZE) -> bytes:
    """Open bytes with Pillow, convert to RGB, resize (keeping aspect) and return JPEG bytes."""
    with Image.open(io.BytesIO(img_bytes)) as img:
        img = img.convert("RGB")
        # Resize preserving aspect ratio
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue()


async def fetch_documents_page(client: httpx.AsyncClient, url: str) -> Dict:
    resp = await client.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


async def list_all_documents(
    client: httpx.AsyncClient, page_size: int = 100, limit: Optional[int] = None
) -> List[Dict]:
    """
    Iterate paginated document list from paperless-ngx.
    This function assumes API returns 'results' and 'next' keys (DRF-style). Adjust if necessary.
    """
    docs = []
    next_url = f"{PAPERLESS_BASE_URL.rstrip('/')}/api/documents/?page_size={page_size}"
    while next_url:
        logger.info(f"Fetching documents page: {next_url}")
        data = await fetch_documents_page(client, next_url)
        page_results = data.get("results") or data.get("data") or data  # tolerant
        if isinstance(page_results, list):
            docs.extend(page_results)
        elif isinstance(page_results, dict):
            # sometimes returned directly as object map
            docs.append(page_results)
        else:
            logger.warning(f"Unexpected documents response shape: {type(page_results)}")
        # Stop if we've reached requested limit
        if limit and len(docs) >= limit:
            return docs[:limit]
        # DRF uses 'next' key; fallback to None
        next_url = data.get("next")
    return docs


async def fetch_thumbnail_bytes(
    client: httpx.AsyncClient, doc_id: int
) -> Optional[bytes]:
    base = PAPERLESS_BASE_URL.rstrip("/")
    # First try thumbnail endpoint (common)
    thumb_url = f"{base}/api/documents/{doc_id}/thumbnail/"
    download_url = f"{base}/api/documents/{doc_id}/download/"
    urls = [thumb_url, download_url]
    for url in urls:
        try:
            logger.debug(f"Trying {url}")
            resp = await client.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                # Some endpoints return JSON with file url; handle that case
                ct = resp.headers.get("content-type", "")
                if "application/json" in ct:
                    j = resp.json()
                    # attempt to find direct file url in JSON
                    # e.g. {"url": "https://.../file.png"}
                    file_url = j.get("url") or j.get("download_url")
                    if file_url:
                        logger.debug(
                            f"Found file URL in JSON response, fetching {file_url}"
                        )
                        resp2 = await client.get(file_url, timeout=REQUEST_TIMEOUT)
                        resp2.raise_for_status()
                        return resp2.content
                    else:
                        logger.debug("JSON response but no file URL field found")
                        continue
                # else treat content as bytes image/pdf
                return resp.content
            else:
                logger.debug(f"Non-200 ({resp.status_code}) when trying {url}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching {url}: {e}")
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
    return None


async def send_image_to_openai(client: httpx.AsyncClient, img_bytes: bytes) -> Dict:
    """
    Send image bytes to OpenAI Responses API for vision analysis.
    This uses an inline base64 data URL. Adjust payload to match the specific model/endpoint you want to use.
    """
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    # Simple prompt asking for OCR + labels
    prompt_text = "Please extract any text (OCR) and list relevant labels and a short caption describing the image."
    payload = {
        "model": OPENAI_MODEL,
        # "input" supports arrays of multimodal blocks for some OpenAI endpoints
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "max_output_tokens": 1000,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    url = "https://api.openai.com/v1/responses"
    logger.info(f"Sending image to OpenAI (model={OPENAI_MODEL})")
    resp = await client.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


async def send_pdf_to_openai(pdf_bytes: bytes, prompt_text: str) -> Dict:
    """
    Upload a PDF using the SDK uploads.upload_file_chunked API (preferred), call the Responses API
    referencing the uploaded file, then delete the uploaded resource. Uses asyncio.to_thread to avoid
    blocking the event loop. This implementation falls back to legacy File.create if the newer uploads
    API is not available in the installed SDK.
    """
    file_id = None
    try:
        # Upload (non-async SDK calls) in a thread to avoid blocking
        def _upload():
            # Use the global OPENAI_CLIENT
            _client = OPENAI_CLIENT

            f = io.BytesIO(pdf_bytes)
            # Some SDK helpers expect a filename attribute on file-like objects
            try:
                f.name = "document.pdf"
            except Exception:
                pass
            # Ensure the stream is at the beginning (defensive)
            try:
                f.seek(0)
            except Exception:
                pass

            # Prefer the newer client.files.create API if present
            if hasattr(_client, "files") and hasattr(_client.files, "create"):
                return _client.files.create(file=f, purpose="user_data")
            # Fallback to chunked upload if available
            if hasattr(_client, "uploads") and hasattr(
                _client.uploads, "upload_file_chunked"
            ):
                return _client.uploads.upload_file_chunked(
                    file=f,
                    mime_type="application/pdf",
                    purpose="user_data",
                )
            # Fallback to legacy File.create
            if hasattr(_client, "File") and hasattr(_client.File, "create"):
                return _client.File.create(
                    file=f, filename="document.pdf", purpose="user_data"
                )
            # As a last resort try module-level File.create
            return openai.File.create(
                file=f, filename="document.pdf", purpose="user_data"
            )

        uploaded = await asyncio.to_thread(_upload)
        # logger.info(uploaded)

        # Extract an ID from common shapes returned by different SDK versions
        file_id = (
            getattr(uploaded, "id", None)
            or getattr(uploaded, "file_id", None)
            or (uploaded.get("id") if isinstance(uploaded, dict) else None)
        )
        logger.info(f"Uploaded PDF to OpenAI, file_id={file_id}")

        # Create a Responses call referencing the uploaded file (run in thread)
        def _create_response():
            # Use the global OPENAI_CLIENT
            _client = OPENAI_CLIENT
            # return _client.completions.create(
            #     model=OPENAI_MODEL,
            #     input=[
            #         {
            #             "role": "user",
            #             "content": [
            #                 {
            #                     "type": "file",
            #                     "file": {
            #                         "file_id": file_id,
            #                     },
            #                 },
            #                 {"type": "text",
            #                  "text": prompt_text},
            #             ],
            #         }
            #     ],
            #     max_output_tokens=1000,
            # )
            #
            #

            response = _client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt_text},
                            {
                                "type": "input_file",
                                "file_id": file_id
                            },
                        ],
                    }
                ],
            )
            return response
        response = await asyncio.to_thread(_create_response)
        return response
    finally:
        if file_id:

            def _delete():
                _client = OPENAI_CLIENT
                # Try several possible deletion entrypoints depending on SDK
                if hasattr(_client, "uploads") and hasattr(_client.uploads, "delete"):
                    try:
                        return _client.uploads.delete(file_id)
                    except Exception:
                        pass
                if hasattr(_client, "files") and hasattr(_client.files, "delete"):
                    try:
                        return _client.files.delete(file_id)
                    except Exception:
                        pass
                if hasattr(_client, "File") and hasattr(_client.File, "delete"):
                    try:
                        return _client.File.delete(file_id)
                    except Exception:
                        pass
                # last attempt: module-level File.delete
                try:
                    return openai.File.delete(file_id)
                except Exception:
                    pass

            try:
                await asyncio.to_thread(_delete)
                logger.info(f"Deleted OpenAI file {file_id} after processing")
            except Exception as e:
                logger.warning(f"Failed to delete OpenAI file {file_id}: {e}")


async def process_document(
    semaphore: asyncio.Semaphore, client: httpx.AsyncClient, doc: Dict
):
    async with semaphore:
        doc_id = doc.get("id") or doc.get("pk") or doc.get("document_id")
        title = doc.get("title") or doc.get("name") or "<untitled>"
        logger.info(f"Processing doc id={doc_id} title={title}")
        if not doc_id:
            logger.warning(f"No id found on document: {doc}")
            return
        thumbnail = await fetch_thumbnail_bytes(client, doc_id)
        if not thumbnail:
            logger.warning(f"No thumbnail/download available for doc {doc_id}")
            return
        # If content is PDF (starts with %PDF) upload the PDF directly to OpenAI for analysis.
        if thumbnail.startswith(b"%PDF"):
            logger.info(
                f"Downloaded content is PDF for doc {doc_id} — uploading to OpenAI"
            )
            prompt_text = "Please extract any text (OCR) and list relevant labels and a short caption describing the document."
            try:
                result = await send_pdf_to_openai(thumbnail, prompt_text)
                # Convert SDK object to serializable dict if possible
                try:
                    if hasattr(result, "to_dict"):
                        result_obj = result.to_dict()
                    elif isinstance(result, dict):
                        result_obj = result
                    else:
                        # Fallback to string representation
                        result_obj = {"raw": str(result)}
                except Exception:
                    result_obj = {"raw": str(result)}
                out_dir = "poc_output"
                os.makedirs(out_dir, exist_ok=True)
                json_out = os.path.join(out_dir, f"{doc_id}.openai.json")
                import json

                with open(json_out, "w", encoding="utf-8") as jf:
                    json.dump(result_obj, jf, ensure_ascii=False, indent=2)
                logger.info(f"Saved OpenAI JSON result to {json_out}")
            except Exception as e:
                logger.exception("Failed to call OpenAI for PDF doc %s: %s", doc_id, e)
            return
        norm = normalize_image_bytes(thumbnail)
        # Optionally save locally for inspection
        out_dir = "poc_output"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{doc_id}.jpg")
        with open(out_path, "wb") as f:
            f.write(norm)
        logger.info(f"Saved normalized image to {out_path}")
        # Send to OpenAI
        try:
            result = await send_image_to_openai(client, norm)
            # Pretty print a subset
            logger.info(f"OpenAI result for doc {doc_id}: {str(result)[:1000]}")
            # Save full JSON for inspection
            json_out = os.path.join(out_dir, f"{doc_id}.openai.json")
            import json

            with open(json_out, "w", encoding="utf-8") as jf:
                json.dump(result, jf, ensure_ascii=False, indent=2)
                logger.info(f"Saved OpenAI JSON result to {json_out}")
        except Exception as e:
            logger.exception("Failed to call OpenAI for doc %s: %s", doc_id, e)


async def main(limit: int = 10, page_size: int = 100):
    headers = {
        "Authorization": f"Token {PAPERLESS_API_TOKEN}"
    }  # paperless-ngx commonly uses Token auth
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        docs = await list_all_documents(client, page_size=page_size, limit=limit)
        logger.info(f"Found {len(docs)} documents (requested limit={limit})")
        semaphore = asyncio.Semaphore(CONCURRENT_WORKERS)
        tasks = []
        for doc in docs:
            tasks.append(process_document(semaphore, client, doc))
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PoC: Fetch paperless-ngx thumbnails -> OpenAI"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of documents to process"
    )
    parser.add_argument(
        "--page_size", type=int, default=100, help="Page size when listing documents"
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(limit=args.limit, page_size=args.page_size))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

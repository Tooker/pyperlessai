import asyncio
import base64
import io
from typing import Optional, Dict, Any

import httpx
from loguru import logger
from PIL import Image
from schemas import DocumentMetadata

# Import openai module for SDK upload fallback in send_pdf_bytes
import openai


class AIClient:
    """
    Encapsulates interactions with OpenAI for image/pdf processing.

    Usage:
        ai = AIClient(api_key="xxx", model="gpt-5")
        async with httpx.AsyncClient(timeout=120) as http_client:
            resp = await ai.send_image_bytes(http_client, img_bytes, prompt_text)
        # For PDFs:
        resp = await ai.send_pdf_bytes(pdf_bytes, prompt_text)
    """

    def __init__(self, api_key: str, model: str = "gpt-5", request_timeout: int = 120):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.model = model
        self.request_timeout = request_timeout

    @staticmethod
    def normalize_image_bytes(img_bytes: bytes, max_size: int = 1024) -> bytes:
        """
        Open bytes with Pillow, convert to RGB, resize (keeping aspect) and return JPEG bytes.
        """
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85)
            return out.getvalue()

    @staticmethod
    def _image_bytes_to_data_url(img_bytes: bytes) -> str:
        """
        Build a base64 data URL for the provided image bytes.

        Attempts to detect the image format via Pillow and sets an appropriate
        MIME type (e.g. image/png, image/jpeg). Falls back to image/jpeg if
        detection fails or the format is uncommon. The data URL uses the raw
        bytes supplied (no conversion) so callers may pass normalized JPEG bytes
        when desired.
        """
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                fmt = (img.format or "").lower()
        except Exception:
            fmt = ""

        if fmt in ("jpeg", "jpg"):
            mime = "image/jpeg"
        elif fmt in ("png", "gif", "webp", "bmp", "tiff", "avif"):
            mime = f"image/{fmt}"
        else:
            mime = "image/jpeg"

        b64 = base64.b64encode(img_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"

    async def send_image_bytes(
        self,
        http_client: httpx.AsyncClient,
        img_bytes: bytes,
        prompt_text: str,
    ) -> Dict[str, Any]:
        """
        Send image bytes to OpenAI Responses API for vision analysis using an inline base64 data URL.
        Uses an externally provided httpx.AsyncClient (so the caller can control headers/timeouts).
        Returns the parsed JSON response.
        """
        # Build a data URL from the provided image bytes using a detected MIME type.
        # This keeps the MIME accurate (e.g. image/png) instead of hardcoding image/jpeg.
        data_url = self._image_bytes_to_data_url(img_bytes)
        payload = {
            "model": self.model,
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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = "https://api.openai.com/v1/responses"
        logger.info(f"Sending image to OpenAI (model={self.model})")
        resp = await http_client.post(url, json=payload, headers=headers, timeout=self.request_timeout)
        resp.raise_for_status()
        return resp.json()

    async def send_pdf_bytes(self, pdf_bytes: bytes, prompt_text: str) -> Dict[str, Any]:
        """
        Upload a PDF using the SDK (synchronously) inside a thread, create a Responses call referencing
        the uploaded file, then delete the uploaded resource. Returns the Responses result object/dict.
        """
        file_id = None

        def _upload_and_create() -> Any:
            # Use openai module and the provided api key
            try:
                # Try to configure the SDK client if it exposes a client class
                # Many SDK installs accept openai.api_key assignment
                try:
                    openai.api_key = self.api_key
                except Exception:
                    pass

                f = io.BytesIO(pdf_bytes)
                try:
                    f.name = "document.pdf"
                except Exception:
                    pass
                try:
                    f.seek(0)
                except Exception:
                    pass

                # Try multiple upload interfaces depending on installed SDK
                client = openai
                uploaded = None
                if hasattr(client, "files") and hasattr(client.files, "create"):
                    uploaded = client.files.create(file=f, purpose="user_data")
                elif hasattr(client, "uploads") and hasattr(client.uploads, "upload_file_chunked"):
                    uploaded = client.uploads.upload_file_chunked(
                        file=f, mime_type="application/pdf", purpose="user_data"
                    )
                elif hasattr(client, "File") and hasattr(client.File, "create"):
                    uploaded = client.File.create(file=f, filename="document.pdf", purpose="user_data")
                else:
                    # Last resort: try module-level File.create
                    uploaded = openai.File.create(file=f, filename="document.pdf", purpose="user_data")

                # Extract file id
                fid = (
                    getattr(uploaded, "id", None)
                    or getattr(uploaded, "file_id", None)
                    or (uploaded.get("id") if isinstance(uploaded, dict) else None)
                )
                if not fid:
                    raise RuntimeError("Failed to obtain uploaded file id from SDK response")
                nonlocal_file_id = fid  # just for readability
                # Create a Responses entry referencing the uploaded file
                # Try multiple ways depending on SDK
                response = None
                if hasattr(client, "responses") and hasattr(client.responses, "create"):
                    response = client.responses.parse(
                        model=self.model,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt_text},
                                    {"type": "input_file", "file_id": fid},
                                ],
                            }
                        ],
                        reasoning={"effort": "low"},
                        text_format=DocumentMetadata
                    )
                elif hasattr(client, "completions") and hasattr(client.completions, "create"):
                    # fallback older API shapes (unlikely but defensive)
                    response = client.completions.create(
                        model=self.model,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "file", "file": {"file_id": fid}},
                                ],
                            }
                        ],
                    )
                else:
                    # If none available raise
                    raise RuntimeError("Installed OpenAI SDK does not expose a recognized responses creation method")

                # Attach file id so caller knows what to clean up
                try:
                    # if response is dict-like, attach
                    if isinstance(response, dict):
                        response["_uploaded_file_id"] = fid
                    else:
                        # some SDK objects allow attribute assignment
                        try:
                            setattr(response, "_uploaded_file_id", fid)
                        except Exception:
                            pass
                except Exception:
                    pass

                return response
            except Exception as e:
                logger.exception("Upload/create via SDK failed: %s", e)
                raise

        # Run the blocking SDK calls in a thread
        response = await asyncio.to_thread(_upload_and_create)

        # Attempt to extract file id for deletion
        if isinstance(response, dict):
            file_id = response.get("_uploaded_file_id")
        else:
            file_id = getattr(response, "_uploaded_file_id", None)

        # Delete uploaded file in background thread if we have an id
        async def _cleanup(fid: str):
            def _delete():
                client = openai
                # try various deletion entrypoints
                try:
                    if hasattr(client, "uploads") and hasattr(client.uploads, "delete"):
                        client.uploads.delete(fid)
                        return
                except Exception:
                    pass
                try:
                    if hasattr(client, "files") and hasattr(client.files, "delete"):
                        client.files.delete(fid)
                        return
                except Exception:
                    pass
                try:
                    if hasattr(client, "File") and hasattr(client.File, "delete"):
                        client.File.delete(fid)
                        return
                except Exception:
                    pass
                try:
                    openai.File.delete(fid)
                except Exception:
                    pass

            try:
                await asyncio.to_thread(_delete)
                logger.info(f"Deleted OpenAI file {fid} after processing")
            except Exception as e:
                logger.warning(f"Failed to delete OpenAI file {fid}: {e}")

        if file_id:
            # Schedule cleanup but don't await here to allow caller to process response faster.
            # Caller may await if they prefer deterministic cleanup.
            asyncio.create_task(_cleanup(file_id))

        return response

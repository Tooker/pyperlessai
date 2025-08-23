from __future__ import annotations

from typing import Optional, Dict, Any, List, Any as AnyType
import asyncio
import json
from pathlib import Path

import httpx
from loguru import logger

from paperless import PaperlessClient
from ai_client import AIClient


class PaperlessOpenAIClassifier:
    """
    Classifier that prefetches Paperless entities (tags, document types, correspondents)
    and sends PDFs to OpenAI with these entities as grounded context inside the prompt.

    This version uses a normal __init__ (no async factory). The actual fetching of
    Paperless entities happens automatically during initialization:

    - If __init__ is called inside a running asyncio event loop and an http_client
      is provided, prefetch runs as a background task. Call `await classifier.ensure_prefetched()`
      to wait for completion when needed.

    - If __init__ is called outside of a running event loop (synchronous context),
      initialization will block and run the prefetch using asyncio.run(...).

    Constructor arguments:
      paperless: PaperlessClient
      ai: AIClient
      http_client: Optional[httpx.AsyncClient] -- if provided, will be used for fetching lists.
      base_prompt: Optional[str] -- override for the base prompt
      prefetch: bool -- whether to fetch lists during init (default True)

    Methods:
      ensure_prefetched() -> awaitable: wait for prefetch to finish (if running in background)
      build_prompt(extra_instructions=None) -> str
      analyze_pdf(pdf_bytes, extra_instructions=None) -> result from AIClient.send_pdf_bytes
      analyze_paperless_document(doc_id, http_client=None, extra_instructions=None) -> fetch & analyze
    """

    def __init__(
        self,
        paperless: PaperlessClient,
        ai: AIClient,
        http_client: Optional[httpx.AsyncClient] = None,
        base_prompt: Optional[str] = None,
        prefetch: bool = True,
    ):
        self.paperless = paperless
        self.ai = ai
        self._external_http_client = http_client
        self.base_prompt = base_prompt or (
            "Analyze the PDF and extract: title, correspondent, tags (max 3), "
            "document_date (YYYY-MM-DD), document_type, language. Prefer values from the allowed lists."
        )

        # placeholders for maps (populated by _prefetch)
        self.tags_map: Dict[int, str] = {}
        self.document_types_map: Dict[int, str] = {}
        self.correspondents_map: Dict[int, str] = {}

        # background prefetch task (if started)
        self._prefetch_task: Optional[asyncio.Task] = None

        if prefetch:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and self._external_http_client:
                # Running inside an event loop and an http client is provided:
                # start background prefetch task (non-blocking). Caller can await ensure_prefetched().
                logger.info("Starting background prefetch task for Paperless entities.")
                self._prefetch_task = loop.create_task(self._prefetch(self._external_http_client))
            else:
                # No running loop (sync context) or no http client provided: run prefetch synchronously.
                # This will create its own AsyncClient if needed and run using asyncio.run.
                logger.info("Running synchronous prefetch for Paperless entities.")
                asyncio.run(self._prefetch(self._external_http_client))

    async def _prefetch(self, http_client: Optional[httpx.AsyncClient] = None) -> None:
        """
        Internal async method to fetch tags, document types and correspondents.
        If http_client is None, a temporary client will be created and closed.
        """
        created_client: Optional[httpx.AsyncClient] = None
        try:
            if http_client is None:
                created_client = httpx.AsyncClient(headers=self.paperless.headers, timeout=self.paperless.timeout)
                http_client = created_client

            logger.info("Fetching tags, document types, and correspondents from Paperless...")
            try:
                self.tags_map = await self.paperless.tags_name_map(http_client)
            except Exception as e:
                logger.exception("Failed to fetch tags: %s", e)
                self.tags_map = {}

            try:
                self.document_types_map = await self.paperless.document_types_name_map(http_client)
            except Exception as e:
                logger.exception("Failed to fetch document types: %s", e)
                self.document_types_map = {}

            try:
                self.correspondents_map = await self.paperless.correspondents_name_map(http_client)
            except Exception as e:
                logger.exception("Failed to fetch correspondents: %s", e)
                self.correspondents_map = {}

            # Ensure deterministic ordering
            self.tags_map = dict(sorted(self.tags_map.items()))
            self.document_types_map = dict(sorted(self.document_types_map.items()))
            self.correspondents_map = dict(sorted(self.correspondents_map.items()))

            logger.info(
                "Prefetch complete: %d tags, %d document types, %d correspondents",
                len(self.tags_map),
                len(self.document_types_map),
                len(self.correspondents_map),
            )
        finally:
            if created_client is not None:
                await created_client.aclose()

    async def ensure_prefetched(self) -> None:
        """
        Await this if a background prefetch task was started and you need to ensure maps are populated.
        Safe to call even if prefetch was run synchronously (no-op).
        """
        if self._prefetch_task:
            await self._prefetch_task
            self._prefetch_task = None

    def _format_allowed_section(self, title: str, mapping: Dict[int, str]) -> str:
        lines = [f"{k}: {v}" for k, v in mapping.items()]
        return f"{title} (id -> name):\n" + ("\n".join(lines) if lines else "(none)")

    def build_prompt(self, extra_instructions: Optional[str] = None) -> str:
        """
        Build the final prompt by loading an external Markdown template
        (prompts/paperless_prompt.md) and substituting the placeholders:

        - {{known_correspondents}} -> JSON array of correspondent names
        - {{tags}} -> JSON array of tag names
        - {{document_types}} -> JSON array of document type names

        The template keeps the {{document_text}} placeholder so callers can insert
        the document content separately (or let the AI client handle it).
        """
        # Prepare allowed lists as arrays of names (preserve order)
        correspondents = list(self.correspondents_map.values())
        tags = list(self.tags_map.values())
        document_types = list(self.document_types_map.values())

        # Attempt to load the external template (relative to project root)
        try:
            template_path = Path(__file__).resolve().parents[1] / "prompts" / "paperless_prompt.md"
            template = template_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.exception("Failed to read prompt template, falling back to inlined prompt: %s", e)
            # Minimal fallback if the external template cannot be read
            template = self.base_prompt

        # Render lists as JSON arrays (preserve unicode characters)
        try:
            known_correspondents_str = json.dumps(correspondents, ensure_ascii=False)
            tags_str = json.dumps(tags, ensure_ascii=False)
            document_types_str = json.dumps(document_types, ensure_ascii=False)
        except Exception:
            known_correspondents_str = str(correspondents)
            tags_str = str(tags)
            document_types_str = str(document_types)

        prompt_text = template.replace("{{known_correspondents}}", known_correspondents_str)
        prompt_text = prompt_text.replace("{{tags}}", tags_str)
        prompt_text = prompt_text.replace("{{document_types}}", document_types_str)

        if extra_instructions:
            prompt_text = "Extra instructions:\n" + extra_instructions + "\n\n" + prompt_text

        return prompt_text

    async def analyze_pdf(self, pdf_bytes: bytes, extra_instructions: Optional[str] = None) -> AnyType:
        """
        Send the provided PDF to OpenAI using AIClient, with a prompt that
        includes grounded context (tags, correspondents, document types).
        Returns the raw response from AIClient.send_pdf_bytes (SDK object or dict).
        """
        prompt_text = self.build_prompt(extra_instructions=extra_instructions)
        logger.info("Sending PDF to OpenAI with grounded context (tags, correspondents, document types).")
        return await self.ai.send_pdf_bytes(pdf_bytes, prompt_text)

    async def analyze_paperless_document(
        self,
        doc_id: int,
        http_client: Optional[httpx.AsyncClient] = None,
        extra_instructions: Optional[str] = None,
    ) -> AnyType:
        """
        Fetch a document from Paperless by id, then analyze it.
        - If it's a PDF, upload to OpenAI with grounded prompt.
        - Otherwise, treat content as image bytes and use vision endpoint.

        If a background prefetch is still running, this method will wait for it to finish
        to ensure grounded lists are available.
        """
        # Wait for prefetch to finish if it's in progress
        await self.ensure_prefetched()

        created_client: Optional[httpx.AsyncClient] = None
        try:
            if http_client is None:
                created_client = httpx.AsyncClient(headers=self.paperless.headers, timeout=self.paperless.timeout)
                http_client = created_client

            content = await self.paperless.fetch_document_bytes(http_client, doc_id)
            if not content:
                raise ValueError(f"No content returned for Paperless document id={doc_id}")

            if content.startswith(b"%PDF"):
                return await self.analyze_pdf(content, extra_instructions=extra_instructions)

            # Image or other bytes: normalize to JPEG and use vision analysis
            # Use a sensible default size; caller can normalize externally if needed.
            norm = AIClient.normalize_image_bytes(content, max_size=1024)
            prompt_text = self.build_prompt(extra_instructions=extra_instructions)
            return await self.ai.send_image_bytes(http_client, norm, prompt_text)
        finally:
            if created_client is not None:
                await created_client.aclose()

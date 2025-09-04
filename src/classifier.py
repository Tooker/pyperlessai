from __future__ import annotations

from typing import Optional, Dict, Any, List, Any as AnyType, Tuple
import asyncio
import json
import os
from pathlib import Path
from copy import deepcopy
from loguru import logger

from paperless import PaperlessClient
from ai_client import AIClient
from schemas import DocumentMetadata, Document
from dotenv import load_dotenv
load_dotenv()

PROCESSED_TAG = os.getenv("PAPERLESS_AI_PROCESSED_TAG")

class PaperlessOpenAIClassifier:
    """
    Classifier that prefetches Paperless entities (tags, document types, correspondents)
    and sends PDFs to OpenAI with these entities as grounded context inside the prompt.

    This version uses a normal __init__ (no async factory). The actual fetching of
    Paperless entities happens automatically during initialization:

    - If __init__ is called inside a running asyncio event loop, prefetch runs as a background task.
      Call `await classifier.ensure_prefetched()` to wait for completion when needed.

    - If __init__ is called outside of a running event loop (synchronous context),
      initialization will block and run the prefetch using asyncio.run(...).

    Constructor arguments:
      paperless: PaperlessClient
      ai: AIClient
      base_prompt: Optional[str] -- override for the base prompt
      prefetch: bool -- whether to fetch lists during init (default True)

    Methods:
      ensure_prefetched() -> awaitable: wait for prefetch to finish (if running in background)
      build_prompt(extra_instructions=None) -> str
      analyze_pdf(pdf_bytes, extra_instructions=None) -> result from AIClient.send_pdf_bytes
      analyze_paperless_document(doc_id, extra_instructions=None) -> fetch & analyze
    """

    def __init__(
        self,
        ai: AIClient,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        request_timeout: int = 30,
        base_prompt: Optional[str] = None,
        prefetch: bool = True,
    ):
        """
        The classifier always creates its own PaperlessClient instance during init.
        - base_url/api_token can be provided; otherwise environment variables are used.
        """
        if ai is None:
            raise ValueError("AIClient instance (ai) is required")
        self.ai = ai

        burl = base_url or os.getenv("PAPERLESS_BASE_URL")
        token = api_token or os.getenv("PAPERLESS_API_TOKEN")
        if not burl or not token:
            raise ValueError("PAPERLESS_BASE_URL and PAPERLESS_API_TOKEN must be provided either as arguments or environment variables")
        self.paperless = PaperlessClient(base_url=burl, api_token=token, request_timeout=request_timeout)

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
        # Remember whether prefetch was requested. Actual prefetching will be
        # started when entering the classifier's async context so the PaperlessClient's
        # shared session can be used for best performance.
        self._prefetch_requested = bool(prefetch)

        # Name of the tag used to mark documents already processed by AI.
        # Configurable via environment variable PAPERLESS_AI_PROCESSED_TAG (fallback: "AI_Processed")
        self.ai_processed_tag_name: str = os.getenv("PAPERLESS_AI_PROCESSED_TAG", "AI_Processed")

    async def _prefetch(self) -> None:
        """
        Internal async method to fetch tags, document types and correspondents.
        Relies on PaperlessClient to manage httpx client lifecycle.
        """
        logger.info("Fetching tags, document types, and correspondents from Paperless...")
        try:
            # Delegate error handling to PaperlessClient; its name-map helpers
            # will return empty dicts and log failures. Keep a top-level catch
            # for any unexpected errors during prefetch.
            self.tags = await self.paperless.list_tag_names()
            self.document_types = await self.paperless.list_document_types_names()
            self.correspondents = await self.paperless.list_correspondents_names()


            logger.info(f"Prefetch complete: {len(self.tags)} tags, {len(self.document_types)} document types, {len(self.correspondents)} correspondents")
        except Exception as e:
            logger.exception(f"Unexpected error during prefetch: {e}")

    async def ensure_prefetched(self) -> None:
        """
        Await this if a background prefetch task was started and you need to ensure maps are populated.
        Safe to call even if prefetch was run synchronously (no-op).
        """
        if self._prefetch_task:
            await self._prefetch_task
            self._prefetch_task = None

    

    async def __aenter__(self) -> "PaperlessOpenAIClassifier":
        """
        Open the internal PaperlessClient shared session and start prefetch (non-blocking)
        if it was requested during initialization.
        """
        # Open shared PaperlessClient session
        await self.paperless.__aenter__()

        # Start prefetch after shared session is active for best performance.
        if self._prefetch_requested:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                logger.info("Starting background prefetch task for Paperless entities (in __aenter__).")
                self._prefetch_task = loop.create_task(self._prefetch())
            else:
                logger.info("Running synchronous prefetch for Paperless entities (in __aenter__).")
                await self._prefetch()

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """
        Ensure any background prefetch finishes and close the internal PaperlessClient session.
        """
        try:
            await self.ensure_prefetched()
        finally:
            await self.paperless.__aexit__(exc_type, exc, tb)

    async def list_documents(self,limit:int) -> List[Document]:
        """
        Convenience wrapper that delegates to the internal PaperlessClient and returns Document models.
        """
        return await self.paperless.list_documents(limit=limit)

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
        
        
        # Attempt to load the external template (relative to project root)
        try:
            template_path = Path(__file__).resolve().parents[1] / "prompts" / "paperless_prompt.md"
            template = template_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.exception(f"Failed to read prompt template, falling back to inlined prompt: {e}")
            # Minimal fallback if the external template cannot be read
            template = self.base_prompt

        prompt_text = template.replace("{{known_correspondents}}", ", ".join(self.correspondents))
        prompt_text = prompt_text.replace("{{tags}}",  ", ".join(self.tags))
        prompt_text = prompt_text.replace("{{document_types}}",  ", ".join(self.document_types))

        if extra_instructions:
            prompt_text = "Extra instructions:\n" + extra_instructions + "\n\n" + prompt_text

        return prompt_text

    async def analyze_pdf(
        self,
        pdf_bytes: bytes,
        extra_instructions: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Send the provided PDF to OpenAI using AIClient, with a prompt that
        includes grounded context (tags, correspondents, document types).

        After receiving a structured response from the AI (DocumentMetadata),
        ensure Paperless contains the referenced correspondent, tags and document_type,
        creating any missing entities. If paperless_doc_id is provided, patch the
        Paperless document to set the correspondent, tags, document_type, title and document_date.

        Returns the raw response from AIClient.send_pdf_bytes (SDK object or dict).
        """
        # Refresh only tags so the grounded prompt uses the latest tag names
        await self._prefetch()
        prompt_text =  self.build_prompt(extra_instructions=extra_instructions)
        logger.debug("Sending PDF to OpenAI with grounded context (tags, correspondents, document types).")
        result = await self.ai.send_pdf_bytes(pdf_bytes, prompt_text)

        # Attempt to extract parsed output that matches DocumentMetadata
       
        try:
            # Common SDK/dict locations
            if hasattr(result, "output_parsed"):
                return result.output_parsed
        except Exception:
            return None


    async def analyze_paperless_document(
        self,
        doc: Document,
        extra_instructions: Optional[str] = None,
        
    ) -> Tuple[Document,bool]:
        """
        Fetch a document from Paperless by id, then analyze it.
        - If it's a PDF, upload to OpenAI with grounded prompt.
        - Otherwise, treat content as image bytes and use vision endpoint.

        If a background prefetch is still running, this method will wait for it to finish
        to ensure grounded lists are available.
        """
        # Wait for prefetch to finish if it's in progress
        localDoc = deepcopy(doc)
        
        result =  await self.analyze_pdf(localDoc.pdfdata, extra_instructions=extra_instructions)
        updateNeeded = result.correspondent !=  localDoc.correspondent or result.tags != localDoc.tags or result.title != localDoc.title or localDoc.document_type != result.document_type or localDoc.document_date != result.document_date
        if len(result.tags) == 0:
            #reject update
            updateNeeded = False
        #update doc
        if updateNeeded:
            localDoc.title = result.title
            localDoc.correspondent = result.correspondent
            localDoc.tags = result.tags
            localDoc.document_type = result.document_type
            localDoc.document_date = result.document_date

        return localDoc,updateNeeded

    async def _process_document(
        self,
        semaphore: asyncio.Semaphore,
        doc: Document
    ) -> None:
        """
        Internal per-document processing previously located in poc.py.
        Fetches document bytes via self.paperless, analyzes with the classifier/AI client,
        and writes JSON output files under out_dir.
        """
        async with semaphore:
            logger.info(f"Analyzing Paperless document Title={doc.title} with classifier")
            try:
                newdoc, updateNeeded = await self.analyze_paperless_document(extra_instructions=None, doc=doc)
                if updateNeeded:
                    logger.info(newdoc)
                    logger.info(doc)
                    if PROCESSED_TAG not in newdoc.tags:
                        newdoc.tags.append(PROCESSED_TAG)
                    
                    await self.paperless.update_document_from_model(newdoc)

                
            except Exception as e:
                logger.exception(f"Failed to call classifier for doc {doc.id}: {e}")
            return

    async def run(
        self,
        limit: int = 10,
        concurrent_workers: int = 4,
        max_image_size: int = 1024,
        out_dir: str = "poc_output",
    ) -> None:
        """
        Convenience method to run the end-to-end flow:
         - open internal PaperlessClient session
         - list documents
         - ensure prefetch
         - process documents concurrently
        """
        # Ensure output directory exists before running
        os.makedirs(out_dir, exist_ok=True)

        async with self:
            docs = await self.list_documents(limit=limit)
            logger.info(f"Found {len(docs)} documents")

            # Wait for classifier prefetch to finish (if it ran in background)
            await self.ensure_prefetched()

            semaphore = asyncio.Semaphore(concurrent_workers)
            tasks = []
            count = 0
            for doc in docs:
                 # If document already processed (has AI_Processed tag), skip it
                if doc.has_ai_processed_tag(PROCESSED_TAG):
                    logger.info(f"Skipping doc title={doc.title} id={doc.id} because it already has AI_Processed tag")
                else:
                    count +=1
                    tasks.append(self._process_document(semaphore, doc))
                    if count == limit:
                        break
            await asyncio.gather(*tasks)
